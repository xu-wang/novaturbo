from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scipy.optimize

from src.ai import geometry19_far_precision_diagnostic as d
from src.ai.geometry19_performance_gated_protocol import ProtocolError, canonical, file_hash
from src.ai.geometry19_performance_gated_physics_protocol import eligibility_flags

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def prohibit_real_physics_and_model_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This suite permits only synthetic evaluators and numerical toy functions")
    monkeypatch.setattr("src.physics.runtime.PhysicsRuntime.evaluate", forbidden)
    monkeypatch.setattr("numpy.load", forbidden)


def test_local_adapter_changes_only_far_and_restores():
    original = d.combustor.brentq
    unrelated = scipy.optimize.brentq
    for arm in ("A", "B"):
        trace = []
        with d.far_adapter(arm, trace):
            assert scipy.optimize.brentq is unrelated
            value, info = d.combustor.brentq(lambda x: x*x-2, 1.0, 2.0,
                xtol=1e-12, rtol=1e-12, maxiter=64, full_output=True, disp=False)
            assert value == pytest.approx(2**.5) and info.converged
        assert d.combustor.brentq is original
        assert trace[0]["xtol"] == d.PARAMETERS[arm]["xtol"]
        assert trace[0]["rtol"] == d.PARAMETERS[arm]["rtol"]
        assert trace[0]["maxiter"] == 64
        assert trace[0]["function_calls"] == info.function_calls


def test_adapter_restores_on_exception_and_rejects_nested_use():
    original = d.combustor.brentq
    with pytest.raises(RuntimeError):
        with d.far_adapter("B", []):
            with pytest.raises(ProtocolError, match="Concurrent"):
                with d.far_adapter("A", []):
                    pass
            raise RuntimeError("synthetic")
    assert d.combustor.brentq is original
    with pytest.raises(ProtocolError, match="Unexpected"):
        with d.far_adapter("B", []):
            d.combustor.brentq(lambda x: x, -1, 1, xtol=1e-10)
    assert d.combustor.brentq is original


@pytest.fixture
def experiment(tmp_path):
    samples = [{"design_hash": str(i), "root_seed_id": "synthetic", "band_index": i//128,
                "split": "validation" if (i%128)//2%5 == 0 else "train",
                "geometry19": {"x": i}, "label_status": "PRELABEL_GEOMETRY_ONLY"} for i in range(512)]
    identities = {arm: {"parent_context_sha256": "ctx", "parent_bundle_sha256": "bundle",
                         "effective_implementation_sha256": arm, "parameters": p} for arm, p in d.PARAMETERS.items()}
    def record(source, arm, identity):
        i = int(source["design_hash"])
        valid = arm == "B" or i >= 83
        row = {**deepcopy(source), "label_status": "V2_2_EVALUATED", "solver": {"converged": True},
               "performance": {"thrust_N": 70.0, "tsfc_kg_N_s": 6e-5, "mass_kg": 1.8},
               "constraint_assessment": {"numerically_valid": valid, "physics_domain_valid": valid},
               "target_domain_assessment": {"inside": True} if valid else None,
               "conservation": {"combustor_energy_W": 1e-9 if valid else 2e-6},
               "pipeline_context_hash": "ctx" if arm == "A" else d.digest({"kind": "diagnostic_context", **identity}),
               "physics_bundle_hash": "bundle" if arm == "A" else identity["effective_implementation_sha256"]}
        row.update(eligibility_flags(row))
        return {"arm": arm, "arm_identity": identity, "evaluation": row, "diagnostics": {}}
    baseline = [record(s, "A", identities["A"])["evaluation"] for s in samples]
    lock = {"samples_sha256": d.digest(samples), "baseline_projection_sha256": d.digest(baseline), "arms": identities}
    d.write_json(tmp_path / "Protocol_Lock.json", lock)
    d.write_once(tmp_path / "Sample_Registry.jsonl", b"".join(canonical(s)+b"\n" for s in samples))
    return tmp_path, lock, samples, baseline, record


def test_complete_budget_order_seal_before_compare_and_zero_call_resume(experiment, monkeypatch):
    output, lock, samples, baseline, fake = experiment
    calls = []
    def evaluator(row, arm, identity):
        if arm == "B":
            assert (output / "A_Reproduction_Gate.json").exists()
        calls.append(arm)
        return fake(row, arm, identity)
    original = d.comparison
    def check(a, b):
        assert (output / "Complete_Manifest.json").is_file()
        return original(a, b)
    monkeypatch.setattr(d, "comparison", check)
    summary = d.execute_locked(output, lock, samples, baseline, evaluator, authorize=True)
    assert calls == ["A"]*512 + ["B"]*512
    assert summary["numerical_issue_resolved"] and summary["recovered_energy_rows"] == 83
    before = {p.name: file_hash(p) for p in output.iterdir()}
    def forbidden(*args):
        pytest.fail("Resume attempted physics")
    assert d.execute_locked(output, lock, samples, baseline, forbidden, resume=True) == summary
    assert before == {p.name: file_hash(p) for p in output.iterdir()}
    assert all(not r["training_candidate"] for r in baseline if r["split"] == "validation")
    (output / "A_Manifest.json").unlink()
    with pytest.raises(ProtocolError, match="missing seals"):
        d.execute_locked(output, lock, samples, baseline, forbidden, resume=True)


def test_a_mismatch_stops_b(experiment):
    output, lock, samples, baseline, fake = experiment
    calls = []
    def mismatch(row, arm, identity):
        calls.append(arm)
        r = fake(row, arm, identity)
        r["evaluation"]["performance"]["mass_kg"] = 1.9
        return r
    summary = d.execute_locked(output, lock, samples, baseline, mismatch, authorize=True)
    assert summary["status"] == "A_REPRODUCTION_FAIL_B_NOT_STARTED"
    assert calls == ["A"]*512 and not list(output.glob("B*"))
    def forbidden(*args):
        pytest.fail("Failed A replayed")
    assert d.execute_locked(output, lock, samples, baseline, forbidden, resume=True) == summary


def test_infrastructure_failure_preserves_orphan_without_retry(experiment):
    output, lock, samples, baseline, fake = experiment
    calls = []
    def crash(*args):
        calls.append(1)
        raise RuntimeError("synthetic infrastructure exception")
    with pytest.raises(RuntimeError):
        d.execute_locked(output, lock, samples, baseline, crash, authorize=True)
    before = {p.name: file_hash(p) for p in output.iterdir()}
    with pytest.raises(ProtocolError, match="Orphan"):
        d.execute_locked(output, lock, samples, baseline, crash, resume=True)
    assert len(calls) == 1 and before == {p.name: file_hash(p) for p in output.iterdir()}


@pytest.mark.parametrize("change", ["order", "split", "count", "arm", "registry"])
def test_changed_input_rejected_before_calls(experiment, change):
    output, lock, samples, baseline, fake = experiment
    if change == "order":
        samples.reverse()
    elif change == "split":
        samples[0]["split"] = "train"
    elif change == "count":
        samples.pop()
    elif change == "arm":
        lock["arms"]["B"]["parameters"] = {"xtol": 1e-9}
    else:
        (output / "Sample_Registry.jsonl").write_text("{}\n")
    def forbidden(*args):
        pytest.fail("Bad input reached evaluator")
    with pytest.raises(ProtocolError):
        d.execute_locked(output, lock, samples, baseline, forbidden, authorize=True)


def test_b_physical_failure_retained_no_retry(experiment):
    output, lock, samples, baseline, fake = experiment
    calls = []
    def physical_failure(row, arm, identity):
        calls.append(arm)
        r = fake(row, arm, identity)
        if arm == "B":
            v = r["evaluation"]
            v["solver"] = {"converged": False, "failure_code": "NO_FLOW_MATCH"}
            v["performance"] = None
            v["conservation"] = {"combustor_energy_W": None}
            v["constraint_assessment"] = {"numerically_valid": False, "physics_domain_valid": False}
            v["target_domain_assessment"] = None
            v.update(eligibility_flags(v))
        return r
    summary = d.execute_locked(output, lock, samples, baseline, physical_failure, authorize=True)
    assert len(calls) == 1024 and summary["counts"]["B"]["train"] == 0
    assert not summary["numerical_issue_resolved"] and not summary["performance_stability_pass"]


def test_real_input_preflight_never_evaluates_or_loads_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight attempted physics or model load")
    monkeypatch.setattr("src.physics.runtime.PhysicsRuntime.evaluate", forbidden)
    monkeypatch.setattr("numpy.load", forbidden)
    rows, baseline, context, lock = d.load_inputs(ROOT)
    assert len(rows) == len(baseline) == 512
    assert not lock["physics_execution_authorized"] and lock["budget"]["total"] == 1024
    assert d.PARAMETERS["A"] == {"xtol": 1e-12, "rtol": 1e-12, "maxiter": 64}


def test_execution_requires_explicit_mode(experiment):
    out, lock, rows, baseline, fake = experiment
    with pytest.raises(ProtocolError, match="Choose"):
        d.execute_locked(out, lock, rows, baseline, fake)


def test_nonfinite_diagnostic_values_keep_list_structure():
    assert d.finite_diagnostics({"calls": [{"x": float("nan"), "ok": True}], "mass": float("inf")}) == {
        "calls": [{"x": None, "ok": True}], "mass": None}


def test_physical_facade_with_synthetic_runtime_and_independent_b_identity(experiment, monkeypatch):
    _, lock, rows, _, fake = experiment
    context = SimpleNamespace(context_hash="ctx", target_domain=object())
    class SyntheticRuntime:
        def evaluate(self, *args):
            d.combustor.brentq(lambda x: x*x-2, 1., 2., xtol=1e-12, rtol=1e-12,
                              maxiter=64, full_output=True, disp=False)
            return SimpleNamespace(operating_point={"fuel_air_ratio": .02, "mass_flow_air_kg_s": 1., "mass_flow_fuel_kg_s": .02})
    context.physics_runtime = SyntheticRuntime()
    def row_evaluator(source, proxy):
        proxy.physics_runtime.evaluate()
        value = fake(source, "A", lock["arms"]["A"])["evaluation"]
        value["physics_model_version"] = "parent_model"
        return value
    monkeypatch.setattr(d, "_evaluate_row", row_evaluator)
    for arm in ("A", "B"):
        record = d.physical_evaluator(rows[100], arm, context, lock["arms"][arm])
        d.validate_record(record, rows[100], arm, lock["arms"][arm])
        assert record["diagnostics"]["fuel_air_ratio"] == .02
        assert record["diagnostics"]["far_solver_invocations"] == 1
        assert record["diagnostics"]["far_calls"][0]["xtol"] == d.PARAMETERS[arm]["xtol"]
        if arm == "B":
            assert record["evaluation"]["parent_pipeline_context_hash"] == "ctx"
            assert record["evaluation"]["physics_model_version"] != "parent_model"


@pytest.mark.parametrize("artifact", ["A-band-0.jsonl", "A-band-0.calls.jsonl", "B-band-3.jsonl", "Summary.json"])
def test_sealed_tamper_rejected_without_replay(experiment, artifact):
    out, lock, rows, baseline, fake = experiment
    d.execute_locked(out, lock, rows, baseline, fake, authorize=True)
    path = out / artifact
    path.write_bytes(path.read_bytes()+b"\n")
    def forbidden(*args):
        pytest.fail("Tampered run attempted new physics")
    with pytest.raises(ProtocolError):
        d.execute_locked(out, lock, rows, baseline, forbidden, resume=True)
