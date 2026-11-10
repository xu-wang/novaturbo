"""WP19-23 target-to-visualization orchestration tests."""

from __future__ import annotations

import json
from hashlib import sha256
import inspect
from pathlib import Path

import pytest
import yaml

import src.ai.geometry19_pipeline as pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "config/inverse_design/targets/wp19_23_smoke_test0_v1.yaml"
V2_2_TARGET_PATH = (
    REPO_ROOT
    / "config/inverse_design/targets/v2_2_development_domain_smoke_v1.yaml"
)


def _install_stage_mocks(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, dict]],
    *,
    physics_stop: bool,
    refinement_stop: bool,
    n_valid: int = 64,
    target_path: Path = TARGET_PATH,
) -> None:
    target_id = pipeline.load_target(target_path).target_id

    def persist(kwargs: dict, summary: dict) -> dict:
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True)
        summary = {**summary, "target_id": target_id}
        encoded = json.dumps(summary, sort_keys=True).encode("utf-8")
        (output / "summary.json").write_bytes(encoded)
        manifest = {
            "target_id": target_id,
            "required_files": ["manifest.json", "summary.json"],
            "summary_sha256": sha256(encoded).hexdigest(),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        return summary

    def candidates(*args, **kwargs):
        calls.append((f"B-{kwargs['mode']}", kwargs))
        return persist(
            kwargs,
            {
                "work_package": "WP19-23B",
                "sampling_mode": kwargs["mode"],
                "candidate_protocol": {"n_valid": n_valid},
                "hashes": {"initial_sources_sha256": "paired-source"},
            },
        )

    def physics(*args, **kwargs):
        calls.append(("C", kwargs))
        return persist(
            kwargs,
            {
                "work_package": "WP19-23C",
                "status": "HIT" if physics_stop else "NO_HIT",
                "downstream_stop": physics_stop,
            },
        )

    def refinement(*args, **kwargs):
        calls.append(("D", kwargs))
        return persist(
            kwargs,
            {
                "work_package": "WP19-23D",
                "status": "HIT" if refinement_stop else "NO_HIT",
                "downstream_stop": refinement_stop,
            },
        )

    def nsga2(*args, **kwargs):
        calls.append(("E", kwargs))
        return persist(
            kwargs,
            {
                "work_package": "WP19-23E",
                "status": "BEST_APPROXIMATION_NOT_HIT",
                "downstream_stop": True,
            },
        )

    def delivery(*args, **kwargs):
        calls.append(("F", kwargs))
        return persist(
            kwargs,
            {
                "work_package": "WP19-23F",
                "status": "DELIVERED",
                "delivered_design_ids": ["a" * 64],
            },
        )

    monkeypatch.setattr(pipeline, "run_candidate_protocol", candidates)
    monkeypatch.setattr(pipeline, "run_true_physics_stage", physics)
    monkeypatch.setattr(pipeline, "run_refinement_stage", refinement)
    monkeypatch.setattr(pipeline, "run_nsga2_stage", nsga2)
    monkeypatch.setattr(pipeline, "run_delivery_stage", delivery)


@pytest.mark.parametrize(
    ("physics_stop", "refinement_stop", "expected", "delivery_source"),
    [
        (True, False, ["B-dflow", "C", "F"], "wp19_23c"),
        (False, True, ["B-dflow", "C", "D", "F"], "wp19_23d"),
        (
            False,
            False,
            ["B-dflow", "C", "D", "E", "F"],
            "wp19_23e",
        ),
    ],
)
def test_pipeline_obeys_hit_triggered_stage_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physics_stop: bool,
    refinement_stop: bool,
    expected: list[str],
    delivery_source: str,
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=physics_stop,
        refinement_stop=refinement_stop,
    )
    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        model_variant="baseline",
        sampling_mode="dflow",
        device="cpu",
        output_root=tmp_path / "run",
    )
    assert [name for name, _ in calls] == expected
    stage_contexts = [kwargs["pipeline_context"] for _, kwargs in calls]
    assert all(context is stage_contexts[0] for context in stage_contexts)
    assert stage_contexts[0].physics_runtime.model_version == "geometry19_closed_v2_1"
    assert Path(calls[-1][1]["source_root"]).name == delivery_source
    assert calls[-1][1]["network_architecture_modified"] is False
    assert summary["status"] == "PASS"
    assert summary["pipeline_profile"] == "v2_1"
    assert summary["pipeline_profile_mode"] == "production_development"
    assert summary["delivered_design_ids"] == ["a" * 64]
    candidate_contract = summary["candidate_filter_contract"]
    assert candidate_contract == {
        "discrete_policy": "unconditional_nearest_integer",
        "discrete_absolute_tolerance_count": None,
        "raw_provenance_version": pipeline.RAW_PROVENANCE_VERSION,
        "dflow_per_candidate_metrics": True,
        "model_bundle_hashes": True,
        "selection": "predicted_rho_ascending_then_candidate_id",
        "hit_alignment": {
            "version": "geometry19_hit_alignment_v1",
            "enabled": True,
            "ordered_outputs": ["thrust_N", "tsfc_kg_N_s", "mass_kg"],
            "axis_scale": "relative_error_divided_by_acceptance_tolerance",
            "hard_metric": "rho_max_axis_ratio",
            "surrogate_loss": "smooth_max_axis_ratio",
            "smooth_max_alpha": 12.0,
            "candidate_ranking": "predicted_rho_ascending_then_candidate_id",
            "true_acceptance": "lexicographic_failed_axes_rho_sum_constraint",
            "legacy_total_true_error": "retained_for_reporting_only",
        },
    }
    stored_contract = json.loads(
        (tmp_path / "run/pipeline_contract.json").read_text(encoding="utf-8")
    )
    assert stored_contract["candidate_filter_contract"] == candidate_contract
    assert stored_contract["pipeline_profile"]["name"] == "v2_1"
    assert stored_contract["profile_registry_version"] == (
        "geometry19_pipeline_profiles_v1"
    )
    assert stored_contract["pipeline_context"]["physics_runtime"][
        "physics_model_version"
    ] == "geometry19_closed_v2_1"
    assert len(
        stored_contract["pipeline_context"]["physics_runtime"][
            "physics_bundle_hash"
        ]
    ) == 64
    aggregate_path = tmp_path / "run/pipeline_manifest.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["manifest_version"] == pipeline.PIPELINE_MANIFEST_VERSION
    assert aggregate["target_id"] == summary["target_id"]
    assert aggregate["stage_order"] == summary["completed_stages"]
    assert aggregate["artifact_count"] == 2 * len(expected)
    assert all(len(row["manifest_sha256"]) == 64 for row in aggregate["stages"])
    assert len(aggregate["hashes"]["constraint_dominance_hash"]) == 64
    assert aggregate["hashes"]["pipeline_profile_hash"] == (
        summary["pipeline_profile_hash"]
    )


def test_pipeline_direct_mode_skips_dflow_but_keeps_c_to_f(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=True,
        refinement_stop=False,
    )
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        model_variant="baseline",
        sampling_mode="direct",
        device="cpu",
        output_root=tmp_path / "run",
    )
    assert [name for name, _ in calls] == ["B-direct", "C", "F"]
    assert calls[1][1]["sampling_mode"] == "direct"


def test_v2_2_development_profile_traverses_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=False,
        refinement_stop=False,
        target_path=V2_2_TARGET_PATH,
    )

    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=V2_2_TARGET_PATH,
        pipeline_profile="v2_2_development",
        model_variant="baseline",
        sampling_mode="direct",
        device="cpu",
        output_root=tmp_path / "v2-2-run",
    )

    assert [name for name, _ in calls] == ["B-direct", "C", "D", "E", "F"]
    assert summary["pipeline_profile"] == "v2_2_development"
    assert summary["pipeline_profile_mode"] == "production_development"
    assert summary["simplified_model"] is True
    assert summary["development_only"] is True
    assert summary["formal_use_allowed"] is False
    assert summary["target_hit"] is False
    assert summary["target_domain_assessment"]["inside"] is True
    contexts = [kwargs["pipeline_context"] for _, kwargs in calls]
    assert all(context is contexts[0] for context in contexts)
    assert contexts[0].physics_runtime.component_maps["compressor"].map_id == (
        "synthetic_centrifugal_compressor_v1"
    )


def test_v2_2_outside_target_domain_stops_before_stage_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=False,
        refinement_stop=False,
    )

    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        pipeline_profile="v2_2_development",
        model_variant="baseline",
        sampling_mode="direct",
        device="cpu",
        output_root=tmp_path / "outside-run",
    )

    assert calls == []
    assert summary["status"] == "OUTSIDE_V2_2_DEVELOPMENT_TARGET_DOMAIN"
    assert summary["completed_stages"] == []
    assert summary["delivery_status"] == "NOT_RUN_OUTSIDE_TARGET_DOMAIN"
    assert summary["target_domain_assessment"]["inside"] is False
    stored_contract = json.loads(
        (tmp_path / "outside-run/pipeline_contract.json").read_text(encoding="utf-8")
    )
    assert stored_contract["target_domain_assessment"] == (
        summary["target_domain_assessment"]
    )
    manifest = json.loads(
        (tmp_path / "outside-run/pipeline_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stage_order"] == []
    assert manifest["artifact_count"] == 0
    assert len(manifest["hashes"]["target_domain_hash"]) == 64


def test_shadow_profile_runs_paired_report_and_forbids_c_to_f(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=False,
        refinement_stop=False,
    )

    def shadow(*args, **kwargs):
        output = Path(args[-1])
        output.mkdir(parents=True)
        summary = {
            "work_package": "WP19-23-SHADOW",
            "status": "PAIRED_PHYSICS_SHADOW_COMPLETE",
            "delivery_enabled": False,
            "formal_use_allowed": False,
        }
        encoded = json.dumps(summary, sort_keys=True).encode("utf-8")
        (output / "summary.json").write_bytes(encoded)
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "WP19-23-SHADOW",
                    "target_id": pipeline.load_target(TARGET_PATH).target_id,
                    "required_files": ["manifest.json", "summary.json"],
                    "summary_sha256": sha256(encoded).hexdigest(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(pipeline, "_run_paired_physics_shadow", shadow)
    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        pipeline_profile="v2_2_shadow",
        model_variant="baseline",
        sampling_mode="direct",
        device="cpu",
        output_root=tmp_path / "shadow-run",
    )

    assert [name for name, _ in calls] == ["B-direct"]
    assert summary["status"] == "PAIRED_PHYSICS_SHADOW_COMPLETE"
    assert summary["completed_stages"] == ["WP19-23B-direct", "WP19-23-SHADOW"]
    assert summary["delivery_enabled"] is False
    assert summary["delivery_status"] == "DISABLED_FOR_SHADOW"
    assert summary["delivered_design_ids"] == []


def test_bounded_pipeline_marks_delivery_architecture_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=True,
        refinement_stop=False,
    )
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        model_variant="bounded-sigmoid",
        device="cpu",
        output_root=tmp_path / "run",
    )
    assert calls[-1][0] == "F"
    assert calls[-1][1]["network_architecture_modified"] is True


def test_zero_valid_candidates_go_directly_to_labeled_cold_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=False,
        refinement_stop=False,
        n_valid=0,
    )
    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        device="cpu",
        output_root=tmp_path / "run",
    )
    assert [name for name, _ in calls] == ["B-dflow", "E", "F"]
    assert calls[1][1]["cold_start"] is True
    assert calls[1][1]["sampling_mode"] == "dflow"
    assert Path(calls[2][1]["source_root"]).name == "wp19_23e"
    assert summary["completed_stages"] == [
        "WP19-23B-dflow",
        "WP19-23E-cold-start",
        "WP19-23F",
    ]


def test_default_pipeline_contract_uses_performance3_and_retains_s0_tag() -> None:
    signature = inspect.signature(pipeline.run_wp19_23_pipeline)
    assert signature.parameters["model_variant"].default is None
    assert signature.parameters["forward_model_tag"].default is None
    assert signature.parameters["sampling_mode"].default is None
    protocol = yaml.safe_load(
        (REPO_ROOT / "config/physics/wp19_23_inverse_closure_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["generation"]["default_sampling_mode"] == "dflow"
    generator = protocol["models"]["generator"]
    assert generator["default_variant"] == "bounded-sigmoid"
    bounded = generator["variants"]["bounded-sigmoid"]
    assert bounded["model_tag"].endswith("bounded_sigmoid_v1")
    assert "bounded_sigmoid_seed43_v1" in bounded["checkpoint"]
    forward = protocol["models"]["forward"]
    assert forward["default_tag"] == "performance3_log"
    assert list(forward["variants"]) == ["performance3_log", "s0"]
    performance = forward["variants"]["performance3_log"]
    assert performance["output_transform"] == "natural_log"
    assert performance["output_contract"] == "performance3_only"
    assert performance["use_constraint_loss"] is False
    assert performance["use_feasibility_loss"] is False
    assert performance["weights"].endswith("performance3_log/seed_43")
    s0 = forward["variants"]["s0"]
    assert s0["model_tag"] == (
        "novaturbo_geometry19_forward_gate_k_s0_legacy_mse_seed43_v1"
    )
    assert s0["weights"].endswith("models/s0_legacy_mse/seed_43")
    assert s0["use_constraint_loss"] is True


def test_omitted_pipeline_defaults_are_resolved_from_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=True, refinement_stop=False
    )
    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        device="cpu",
        output_root=tmp_path / "run",
    )
    assert summary["model_variant"] == "bounded-sigmoid"
    assert summary["forward_model_tag"] == "performance3_log"
    assert summary["sampling_mode"] == "dflow"
    assert calls[0][0] == "B-dflow"
    assert calls[0][1]["forward_model_tag"] == "performance3_log"
    stored = json.loads(
        (tmp_path / "run/pipeline_contract.json").read_text(encoding="utf-8")
    )
    assert stored["generator_model_tag"].endswith("bounded_sigmoid_v1")
    assert stored["forward_model"]["logical_tag"] == "performance3_log"
    assert stored["forward_model"]["output_transform"] == "natural_log"
    assert stored["forward_model"]["use_constraint_loss"] is False
    assert "bounded_sigmoid_seed43_v1" in stored["generator_checkpoint"]
    assert len(stored["generator_normalizer_hash"]) == 64


def test_pipeline_s0_tag_selects_previous_forward_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=False, refinement_stop=True
    )
    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        forward_model_tag="s0",
        device="cpu",
        output_root=tmp_path / "run",
    )
    assert summary["forward_model_tag"] == "s0"
    assert calls[0][1]["forward_model_tag"] == "s0"
    assert calls[2][1]["forward_model_tag"] == "s0"
    stored = json.loads(
        (tmp_path / "run/pipeline_contract.json").read_text(encoding="utf-8")
    )
    assert stored["forward_model"]["logical_tag"] == "s0"
    assert stored["forward_model"]["output_contract"] == (
        "performance3_plus_constraint_residuals"
    )
    assert stored["forward_model"]["use_constraint_loss"] is True


def test_pipeline_rejects_checkpoint_tag_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=True, refinement_stop=False
    )
    original = pipeline.run_candidate_protocol

    def candidates_with_wrong_tag(*args, **kwargs):
        summary = original(*args, **kwargs)
        summary["model_hashes"] = {
            "generator": {"model_tag": "wrong_generator_tag"}
        }
        return summary

    monkeypatch.setattr(pipeline, "run_candidate_protocol", candidates_with_wrong_tag)
    with pytest.raises(pipeline.Geometry19PipelineError, match="model_tag differs"):
        pipeline.run_wp19_23_pipeline(
            REPO_ROOT,
            target_path=TARGET_PATH,
            device="cpu",
            output_root=tmp_path / "run",
        )


def test_pipeline_forwards_export_mesh_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch,
        calls,
        physics_stop=True,
        refinement_stop=False,
    )
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        export_mesh=True,
        device="cpu",
        output_root=tmp_path / "run",
    )
    assert calls[-1][0] == "F"
    assert calls[-1][1]["export_mesh"] is True


def test_pipeline_sync_marker_is_present_in_cli_and_orchestrator() -> None:
    cli = (REPO_ROOT / "scripts/run_wp19_23_pipeline.py").read_text(encoding="utf-8")
    module = (REPO_ROOT / "src/ai/geometry19_pipeline.py").read_text(encoding="utf-8")
    assert pipeline.PIPELINE_SYNC_MARKER in cli
    assert pipeline.PIPELINE_SYNC_MARKER in module
    assert "--restart-from-stage" in cli
    assert pipeline.PIPELINE_STAGE_ORDER == (
        "WP19-23B-single-generation-direct-or-dflow",
        "WP19-23C",
        "WP19-23D-conditional",
        "WP19-23E-conditional",
        "WP19-23F",
    )


def test_inline_target_values_are_canonical_and_positive() -> None:
    value = pipeline.build_target_mapping(
        thrust_N=16.0,
        tsfc_kg_N_s=1.5e-4,
        mass_kg=2.0,
    )
    assert len(value["target_id"]) == 64
    with pytest.raises(Exception, match="finite and positive"):
        pipeline.build_target_mapping(
            thrust_N=0.0,
            tsfc_kg_N_s=1.5e-4,
            mass_kg=2.0,
        )


def test_resume_rejects_sampling_mode_drift_before_reusing_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=True, refinement_stop=False
    )
    runtime = tmp_path / "run"
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        sampling_mode="dflow",
        device="cpu",
        output_root=runtime,
    )
    with pytest.raises(pipeline.Geometry19PipelineError, match="Resume arguments differ"):
        pipeline.run_wp19_23_pipeline(
            REPO_ROOT,
            target_path=TARGET_PATH,
            sampling_mode="direct",
            device="cpu",
            output_root=runtime,
            resume=True,
        )


def test_stage_artifact_audit_rejects_summary_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=True, refinement_stop=False
    )
    runtime = tmp_path / "run"
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        device="cpu",
        output_root=runtime,
    )
    stage = runtime / "wp19_23c"
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    summary["status"] = "TAMPERED"
    (stage / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(pipeline.Geometry19PipelineError, match="summary hash mismatch"):
        pipeline._stage_artifact_entries(runtime, stage)


def test_resume_rejects_tampered_summary_before_stage_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=True, refinement_stop=False
    )
    runtime = tmp_path / "run"
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        device="cpu",
        output_root=runtime,
    )
    summary_path = runtime / "wp19_23c/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "TAMPERED"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(pipeline.Geometry19PipelineError, match="summary hash mismatch"):
        pipeline.run_wp19_23_pipeline(
            REPO_ROOT,
            target_path=TARGET_PATH,
            device="cpu",
            output_root=runtime,
            resume=True,
        )


def test_restart_from_f_validates_and_reuses_upstream_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=False, refinement_stop=False
    )
    runtime = tmp_path / "run"
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        device="cpu",
        output_root=runtime,
    )
    (runtime / "wp19_23f/old-marker.txt").write_text("old", encoding="utf-8")
    calls.clear()

    summary = pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        device="cpu",
        output_root=runtime,
        restart_from_stage="F",
    )

    assert [name for name, _ in calls] == ["F"]
    assert not (runtime / "wp19_23f/old-marker.txt").exists()
    assert summary["completed_stages"] == [
        "WP19-23B-dflow",
        "WP19-23C",
        "WP19-23D",
        "WP19-23E",
        "WP19-23F",
    ]


def test_restart_from_f_rejects_tampered_upstream_before_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    _install_stage_mocks(
        monkeypatch, calls, physics_stop=False, refinement_stop=False
    )
    runtime = tmp_path / "run"
    pipeline.run_wp19_23_pipeline(
        REPO_ROOT,
        target_path=TARGET_PATH,
        device="cpu",
        output_root=runtime,
    )
    marker = runtime / "wp19_23f/old-marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    (runtime / "wp19_23e/summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(pipeline.Geometry19PipelineError, match="summary hash mismatch"):
        pipeline.run_wp19_23_pipeline(
            REPO_ROOT,
            target_path=TARGET_PATH,
            device="cpu",
            output_root=runtime,
            restart_from_stage="F",
        )

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_restart_from_stage_rejects_unsupported_or_conflicting_modes(
    tmp_path: Path,
) -> None:
    with pytest.raises(pipeline.Geometry19PipelineError, match="Only restart"):
        pipeline.run_wp19_23_pipeline(
            REPO_ROOT,
            target_path=TARGET_PATH,
            output_root=tmp_path / "missing",
            restart_from_stage="E",
        )
    with pytest.raises(pipeline.Geometry19PipelineError, match="not both"):
        pipeline.run_wp19_23_pipeline(
            REPO_ROOT,
            target_path=TARGET_PATH,
            output_root=tmp_path / "missing",
            resume=True,
            restart_from_stage="F",
        )


def test_stage_artifact_audit_rejects_missing_and_unsafe_required_paths(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "run"
    stage = runtime / "stage"
    stage.mkdir(parents=True)
    (stage / "summary.json").write_text("{}", encoding="utf-8")
    (stage / "manifest.json").write_text(
        json.dumps({"required_files": ["missing.json"], "summary_sha256": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(pipeline.Geometry19PipelineError, match="artifact is missing"):
        pipeline._stage_artifact_entries(runtime, stage)
    (stage / "manifest.json").write_text(
        json.dumps({"required_files": ["../escape.json"], "summary_sha256": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(pipeline.Geometry19PipelineError, match="Unsafe required"):
        pipeline._stage_artifact_entries(runtime, stage)
