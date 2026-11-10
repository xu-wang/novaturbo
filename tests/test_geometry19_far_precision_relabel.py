from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import json

import pytest
import scipy.optimize

from src.ai import geometry19_far_precision_relabel as m
from src.physics import geometry19_far_precision_v1 as backend
from src.ai.geometry19_performance_gated_protocol import ProtocolError, file_hash
from src.ai.geometry19_performance_gated_physics_protocol import eligibility_flags

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_real_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Only synthetic physics permitted in this test suite")
    monkeypatch.setattr("src.physics.runtime.PhysicsRuntime.evaluate", forbidden)
    monkeypatch.setattr("numpy.load", forbidden)


def test_isolated_function_globals_and_parameters():
    original = backend.original.solve_combustor_map_enhanced
    global_solver = scipy.optimize.brentq
    for precision in (False, True):
        counts = {"far_solver_invocations": 0, "far_function_evaluations": 0}
        fn = backend.isolated_combustor(counts, precision=precision)
        assert fn.__code__ is original.__code__ and fn.__globals__ is not original.__globals__
        assert {k for k in fn.__globals__ if fn.__globals__[k] is not original.__globals__[k]} == {"brentq"}
        value, info = fn.__globals__["brentq"](lambda x:x*x-2, 1., 2., xtol=1e-12, rtol=1e-12,
                                               maxiter=64, full_output=True, disp=False)
        assert value == pytest.approx(2**.5) and counts["far_function_evaluations"] == info.function_calls
        assert backend.original.brentq is global_solver and scipy.optimize.brentq is global_solver
        with pytest.raises(ProtocolError):
            fn.__globals__["brentq"](lambda x:x, -1., 1., xtol=1e-9)


def test_real_inputs_recursive_seals_and_new_identity_without_physics():
    i = m.load(ROOT)
    assert len(i.rows) == 16896 and len(i.r1) == 512 and len(i.r2) == 128
    assert i.candidate.context_hash != i.parent.context_hash
    assert i.candidate.physics_runtime.physics_bundle_hash != i.parent.physics_runtime.physics_bundle_hash
    assert i.candidate.physics_runtime.backend_id != i.parent.physics_runtime.backend_id
    assert i.candidate.physics_runtime.model_version == backend.VERSION
    assert i.candidate.target_domain is i.parent.target_domain
    assert i.candidate.physics_runtime.conservation_tolerances is i.parent.physics_runtime.conservation_tolerances
    # Call backend injection only with a fake whole-engine solver; never real physics.
    seen = {}
    def fake_solver(*args, **kwargs):
        seen.update(kwargs)
        return "synthetic"
    fake_parent = replace(i.parent, physics_runtime=replace(i.parent.physics_runtime,
        backend=replace(i.parent.physics_runtime.backend, solver=fake_solver)))
    candidate, _, _ = backend.build_candidate(fake_parent, ROOT)
    supplied = partial(backend.original.solve_combustor_map_enhanced, component_map=object(), thermo=object(), map_policy="diagnostic")
    sentinel = object()
    assert candidate.physics_runtime.backend.solver(combustor_solver=supplied, nozzle_solver=sentinel) == "synthetic"
    assert seen["nozzle_solver"] is sentinel and seen["combustor_solver"].func is not supplied.func
    assert seen["combustor_solver"].keywords == supplied.keywords


def synthetic_row(source, context):
    row = {**source, "label_status": "V2_2_EVALUATED", "solver": {"converged": True},
           "constraint_assessment": {"numerically_valid": True, "physics_domain_valid": True},
           "performance": {"thrust_N": 70., "tsfc_kg_N_s": 6e-5, "mass_kg": 1.8},
           "target_domain_assessment": {"inside": True}, "conservation": {"combustor_energy_W": 1e-10},
           "map_states": {}, "physics_model_version": context.physics_runtime.model_version,
           "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
           "pipeline_context_hash": context.context_hash, "target_domain_hash": "domain"}
    row.update(eligibility_flags(row))
    return row


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    parent = SimpleNamespace(context_hash="old", physics_runtime=SimpleNamespace(model_version="old", physics_bundle_hash="old"),
                             target_domain=SimpleNamespace(source_hash="domain"))
    candidate = SimpleNamespace(context_hash="new", physics_runtime=SimpleNamespace(model_version="new", physics_bundle_hash="new"),
                                target_domain=SimpleNamespace(source_hash="domain"))
    r1 = [{"design_hash": str(j), "band_index": j//128, "split": "train", "label_status": "PRELABEL_GEOMETRY_ONLY"} for j in range(512)]
    r2 = [{"design_hash": str(j), "band_index": j%4, "split": "train", "label_status": "PRELABEL_GEOMETRY_ONLY"} for j in range(512,640)]
    i = SimpleNamespace(r1=r1, r2=r2, rows=r1+r2, old=[synthetic_row(r,parent) for r in r1+r2],
                        b=[synthetic_row(r,candidate) for r in r1], parent=parent, candidate=candidate, counters={}, config={})
    m.write_json(tmp_path / "Protocol_Lock.json", {"synthetic": True})
    monkeypatch.setattr(m, "tracked_parent", lambda p: (p,{}))
    return tmp_path, i


def test_regression_budget_order_and_zero_call_resume(inputs):
    out, i = inputs
    calls = []
    def candidate(row, ctx):
        if len(calls) >= 512:
            assert (out / "R2_A_Gate.json").exists()
        calls.append("C")
        return synthetic_row(row,ctx)
    def parent(row,ctx):
        assert (out / "R1_Gate.json").exists()
        calls.append("A")
        return synthetic_row(row,ctx)
    summary = m.execute_stages(out,i,"regression",candidate_eval=candidate,parent_eval=parent)
    assert summary["status"] == "REGRESSION_PASS" and calls == ["C"]*512+["A"]*128+["C"]*128
    before = {str(p):file_hash(p) for p in out.rglob("*") if p.is_file()}
    def forbidden(*args):
        pytest.fail("Resume called physics")
    assert m.execute_stages(out,i,"regression",resume=True,candidate_eval=forbidden,parent_eval=forbidden) == summary
    assert before == {str(p):file_hash(p) for p in out.rglob("*") if p.is_file()}
    (out / "R2_C/Manifest.json").unlink()
    with pytest.raises(ProtocolError,match="seals missing"):
        m.execute_stages(out,i,"regression",resume=True,candidate_eval=forbidden,parent_eval=forbidden)


@pytest.mark.parametrize("bad_stage",["R1","R2_A","R2_C"])
def test_regression_failure_stops_and_forbids_relabel(inputs,bad_stage):
    out,i = inputs
    calls = []
    def evaluate(row,ctx):
        name = "R1" if len(calls)<512 else "R2_A" if len(calls)<640 else "R2_C"
        calls.append(name)
        r=synthetic_row(row,ctx)
        if name==bad_stage:
            r["performance"]["mass_kg"]=2.0
        return r
    s=m.execute_stages(out,i,"regression",candidate_eval=evaluate,parent_eval=evaluate)
    assert s["status"]=="REGRESSION_FAIL_"+bad_stage
    assert len(calls)=={"R1":512,"R2_A":640,"R2_C":768}[bad_stage]
    with pytest.raises(ProtocolError,match="Regression not passed"):
        m.execute_stages(out,i,"relabel",candidate_eval=evaluate,parent_eval=evaluate)
    assert not (out / "Relabel_Execution_Manifest.json").exists()


def test_orphan_preserved_no_retry(inputs):
    out,i=inputs
    calls=[]
    def crash(*args):
        calls.append(1)
        raise RuntimeError("synthetic infrastructure error")
    with pytest.raises(RuntimeError):
        m.execute_stages(out,i,"regression",candidate_eval=crash,parent_eval=crash)
    before={str(p):file_hash(p) for p in out.rglob("*") if p.is_file()}
    with pytest.raises(ProtocolError,match="Orphan"):
        m.execute_stages(out,i,"regression",resume=True,candidate_eval=crash,parent_eval=crash)
    assert len(calls)==1 and before=={str(p):file_hash(p) for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize("reason",["energy","solver","eligibility","performance","missing"])
def test_migration_gate_catches_regression(inputs,reason):
    _,i=inputs
    old=[i.old[0]]
    new=deepcopy(old)
    r=new[0]
    if reason=="energy":r["conservation"]["combustor_energy_W"]=2e-6
    elif reason=="solver":r["solver"]["converged"]=False
    elif reason=="eligibility":r["physics_eligible"]=False
    elif reason=="performance":r["performance"]["thrust_N"]=71.
    else:r["performance"]=None
    assert not m.migration_gate(old,new)["pass"]


def test_missing_original_performance_is_not_imputed(inputs):
    _,i=inputs
    old=deepcopy(i.old[:1]);old[0]["performance"]=None
    result=m.migration_gate(old,i.old[:1])
    assert result["pass"] and len(result["incomparable"])==3


def test_full_relabel_seals_before_gates_no_reuse_and_completed_resume(inputs,monkeypatch):
    out,i=inputs
    rows=[{"design_hash":str(j),"root_seed_id":str(j//512),"band_index":j%512//128,
           "split":"validation" if (j%128)//2%5==0 else "train", "label_status":"PRELABEL_GEOMETRY_ONLY"} for j in range(16896)]
    i.rows=rows;i.r1=rows[:512];i.r2=rows[512:640]
    i.old=[synthetic_row(r,i.parent) for r in rows]
    i.b=[synthetic_row(r,i.candidate) for r in i.r1]
    i.config={"relabel":{"gates":{"each_root_minimum_valid_train":256,"each_root_minimum_valid_validation":64,
                                    "each_root_band_minimum_valid_train":32,"each_root_band_minimum_valid_validation":8}}}
    calls=[]
    def evaluate(row,ctx):
        calls.append(row["design_hash"])
        return synthetic_row(row,ctx)
    assert m.execute_stages(out,i,"regression",candidate_eval=evaluate,parent_eval=evaluate)["status"]=="REGRESSION_PASS"
    original_gate=m.migration_gate
    def gate(old,new):
        if len(old)==16896:
            assert (out/"L/Manifest.json").exists() and (out/"L/Sealed_Evaluations.jsonl").exists()
        return original_gate(old,new)
    monkeypatch.setattr(m,"migration_gate",gate)
    summary=m.execute_stages(out,i,"relabel",candidate_eval=evaluate,parent_eval=evaluate)
    assert summary["status"]=="RELABEL_PASS" and summary["valid_train"]==13464 and summary["valid_validation"]==3432
    assert len(calls)==17664 and calls[768:]==[r["design_hash"] for r in rows]
    before={str(p):file_hash(p) for p in out.rglob("*") if p.is_file()}
    def forbidden(*args):pytest.fail("Sealed relabel replayed physics")
    assert m.execute_stages(out,i,"relabel",resume=True,candidate_eval=forbidden,parent_eval=forbidden)==summary
    assert before=={str(p):file_hash(p) for p in out.rglob("*") if p.is_file()}
    (out/"L/shards/part-00131.jsonl").unlink()
    with pytest.raises(ProtocolError):
        m.execute_stages(out,i,"relabel",resume=True,candidate_eval=forbidden,parent_eval=forbidden)


@pytest.mark.parametrize("change",["data","journal","execution"])
def test_tampering_refused_without_new_calls(inputs,change):
    out,i=inputs
    m.execute_stages(out,i,"regression",candidate_eval=synthetic_row,parent_eval=synthetic_row)
    path=out/{"data":"R1/shards/part-00000.jsonl","journal":"R1/shards/part-00000.calls.jsonl","execution":"Regression_Execution_Manifest.json"}[change]
    path.write_bytes(path.read_bytes()+b"\n")
    def forbidden(*args):pytest.fail("Tampered run reached evaluator")
    with pytest.raises(ProtocolError):
        m.execute_stages(out,i,"regression",resume=True,candidate_eval=forbidden,parent_eval=forbidden)
