"""Integration gates for the non-deliverable V2.2 synthetic-map shadow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_pipeline import _run_paired_physics_shadow
from src.physics import Environment, Geometry19


REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def test_shadow_runtime_has_hashed_map_and_thermo_dependencies() -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_shadow")
    runtime = context.physics_runtime

    assert context.profile.mode == "paired_physics_shadow"
    assert context.protocol["delivery"]["enabled"] is False
    assert context.protocol["shadow"]["forbidden_stages"] == ["D", "E", "F"]
    assert runtime.backend_id == "geometry19_v2_2_shadow_backend"
    assert runtime.model_version == "geometry19_map_enhanced_v2_2_shadow"
    assert runtime.thermo_model is not None
    assert set(runtime.component_maps) == {"compressor"}
    assert len(runtime.provenance_hashes["thermo_model_hash"]) == 64
    assert len(runtime.provenance_hashes["compressor_map_hash"]) == 64
    assert runtime.physics_bundle_hash != build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_1"
    ).physics_runtime.physics_bundle_hash


def test_shadow_reference_design_keeps_one_dimensional_root_and_conservation() -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_shadow")
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())

    assert result.physics_model_version == "geometry19_map_enhanced_v2_2_shadow"
    assert result.solver.converged is True
    assert result.operating_point["flow_root_count"] == 1
    assert result.operating_point["selected_flow_root_index"] == 0
    compressor = result.components["compressor"]
    assert compressor["map_id"] == "synthetic_centrifugal_compressor_v1"
    assert compressor["map_in_domain"] is True
    assert compressor["map_extrapolation_distance"] == 0.0
    assert 0.0 <= compressor["surge_margin"] <= 1.0
    assert 0.0 <= compressor["choke_margin"] <= 1.0
    tolerances = context.physics_runtime.conservation_tolerances
    assert abs(result.conservation.combustor_energy_W) <= float(
        tolerances["combustor_energy_absolute_W"]
    )
    assert abs(result.conservation.shaft_power_W) <= float(
        tolerances["shaft_power_absolute_W"]
    )
    assert abs(result.conservation.normalized_flow) <= float(
        tolerances["normalized_flow_absolute"]
    )
    assert abs(result.conservation.nozzle_velocity_m_s) <= float(
        tolerances["nozzle_velocity_absolute_m_s"]
    )


def test_hecc_shadow_profile_is_isolated_and_non_deliverable() -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_hecc_shadow")
    runtime = context.physics_runtime

    assert context.profile.mode == "paired_physics_shadow"
    assert context.protocol["delivery"]["enabled"] is False
    assert context.protocol["shadow"]["forbidden_stages"] == ["D", "E", "F"]
    assert runtime.component_maps["compressor"].map_id == (
        "nasa_hecc_post_test_cfd_v1"
    )
    assert runtime.component_maps["compressor"].formal_use_allowed is False
    assert runtime.physics_bundle_hash != build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_shadow"
    ).physics_runtime.physics_bundle_hash

    result = runtime.evaluate(_reference_design(context), Environment())
    assert result.solver.converged is True
    assert result.operating_point["flow_root_count"] == 1
    compressor = result.components["compressor"]
    assert compressor["map_id"] == "nasa_hecc_post_test_cfd_v1"
    assert 0.7 < compressor["work_coefficient"] < 0.9
    assert compressor["map_in_domain"] is True
    assert compressor["choke_margin"] < 0.0


def test_v2_1_remains_default_and_contains_no_map_diagnostics() -> None:
    context = build_pipeline_context(REPO_ROOT)
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())

    assert context.profile.name == "v2_1"
    assert result.physics_model_version == "geometry19_closed_v2_1"
    assert "map_id" not in result.components["compressor"]


def test_paired_shadow_artifacts_preserve_candidate_identity_and_never_deliver(
    tmp_path: Path,
) -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_shadow")
    design = _reference_design(context)
    generation = tmp_path / "generation"
    generation.mkdir()
    candidate = {
        "candidate_id": 7,
        "design_hash": "a" * 64,
        "geometry19": design.as_mapping(),
        "valid": True,
    }
    (generation / "candidate_filter_ledger.jsonl").write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    summary = _run_paired_physics_shadow(
        REPO_ROOT,
        generation,
        {
            "target_id": "shadow-test",
            "candidate_protocol": {"kept_candidate_ids": [7]},
        },
        context,
        tmp_path / "shadow",
    )

    assert summary["paired_candidate_count"] == 1
    assert summary["delivery_enabled"] is False
    paired = json.loads(
        (tmp_path / "shadow/paired_physics_ledger.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert paired["candidate_id"] == 7
    assert paired["baseline"]["design_hash"] == paired["enhanced"]["design_hash"]
    assert paired["baseline"]["physics_model_version"] == "geometry19_closed_v2_1"
    assert paired["enhanced"]["physics_model_version"] == (
        "geometry19_map_enhanced_v2_2_shadow"
    )
    assert paired["enhanced"]["compressor"]["map_in_domain"] is True
    assert len(summary["component_map_registry_hash"]) == 64
