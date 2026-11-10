"""Regression contract for the single Geometry19 V2.1 physics runtime."""

from __future__ import annotations

from dataclasses import asdict
import inspect
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_pipeline_context import (
    Geometry19PipelineContextError,
    build_pipeline_context,
    resolve_pipeline_context,
)
from src.ai.geometry19_pipeline_profiles import (
    Geometry19PipelineProfileError,
    load_pipeline_profile_registry,
)
from src.physics import Environment, Geometry19, solve_design_point
from src.visualization import Geometry19DesignStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def test_v2_1_runtime_is_exact_solver_adapter() -> None:
    context = build_pipeline_context(REPO_ROOT)
    runtime = context.physics_runtime
    design = _reference_design(context)
    environment = Environment()
    direct = solve_design_point(
        design,
        environment,
        runtime.model_config,
        operating_point_policy=runtime.operating_point_policy,
    )
    through_runtime = runtime.evaluate(design, environment)

    assert asdict(through_runtime) == asdict(direct)
    assert runtime.backend_id == "geometry19_v2_1_backend"
    assert runtime.model_version == "geometry19_closed_v2_1"
    assert len(runtime.physics_bundle_hash) == 64
    assert context.provenance_hashes["physics_bundle_hash"] == runtime.physics_bundle_hash
    assert context.profile.name == "v2_1"
    assert context.profile.mode == "production_development"
    assert len(context.profile.profile_hash) == 64


def test_profile_registry_defaults_to_v2_1_and_registers_shadow() -> None:
    registry = load_pipeline_profile_registry(REPO_ROOT)

    assert registry.registry_version == "geometry19_pipeline_profiles_v1"
    assert registry.default_profile == "v2_1"
    assert registry.profile_names == (
        "v2_1",
        "v2_2_shadow",
        "v2_2_hecc_shadow",
        "v2_2_comp_turb_shadow",
        "v2_2_comp_turb_inlet_shadow",
        "v2_2_comp_turb_inlet_comb_shadow",
        "v2_2_full_component_shadow",
        "v2_2_full_variable_thermo_shadow",
        "v2_2_development",
    )
    selected = registry.resolve()
    assert selected.name == "v2_1"
    assert selected.protocol_path == (
        REPO_ROOT / "config/physics/wp19_23_inverse_closure_v1.yaml"
    ).resolve()
    shadow = registry.resolve("v2_2_shadow")
    assert shadow.mode == "paired_physics_shadow"
    with pytest.raises(Geometry19PipelineProfileError, match="Unknown pipeline profile"):
        registry.resolve("v3_engineering")


def test_profile_registry_rejects_unknown_execution_mode(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("placeholder: true\n", encoding="utf-8")
    registry = tmp_path / "profiles.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "registry_version": "test",
                "default_profile": "bad",
                "profiles": {
                    "bad": {
                        "protocol": "protocol.yaml",
                        "mode": "paired_physics_shdow",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Geometry19PipelineProfileError, match="unsupported mode"):
        load_pipeline_profile_registry(tmp_path, registry_path=registry)


def test_explicit_default_profile_has_the_same_context_identity() -> None:
    implicit = build_pipeline_context(REPO_ROOT)
    explicit = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_1")

    assert explicit.context_hash == implicit.context_hash
    assert explicit.physics_runtime.physics_bundle_hash == (
        implicit.physics_runtime.physics_bundle_hash
    )
    assert explicit.contract_mapping()["pipeline_profile"]["name"] == "v2_1"


def test_v2_2_development_uses_full_synthetic_chain_without_formal_claims() -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_development")
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())

    assert context.profile.mode == "production_development"
    assert context.protocol["delivery"]["enabled"] is True
    assert context.protocol["delivery"]["development_only"] is True
    assert context.protocol["physics_evaluation"]["formal_use_allowed"] is False
    assert context.physics_runtime.component_maps["compressor"].map_id == (
        "synthetic_centrifugal_compressor_v1"
    )
    assert all(
        component_map.formal_use_allowed is False
        for component_map in context.physics_runtime.component_maps.values()
    )
    assert result.solver.converged is True
    assert context.target_domain is not None
    assert context.target_domain.domain_version == (
        "geometry19_v2_2_development_target_domain_v1"
    )
    assert context.provenance_hashes["target_domain_hash"] == (
        context.target_domain.source_hash
    )
    assert context.contract_mapping()["target_domain"]["sha256"] == (
        context.target_domain.source_hash
    )


def test_v2_1_context_has_no_target_domain_preflight() -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_1")

    assert context.target_domain is None
    assert context.contract_mapping()["target_domain"] is None
    assert "target_domain_hash" not in context.provenance_hashes


def test_context_rejects_repository_and_solver_overrides() -> None:
    context = build_pipeline_context(REPO_ROOT)
    with pytest.raises(Geometry19PipelineContextError, match="repository root"):
        resolve_pipeline_context(REPO_ROOT.parent, context)
    with pytest.raises(Geometry19PipelineContextError, match="override the solver"):
        resolve_pipeline_context(REPO_ROOT, context, solver=solve_design_point)
    with pytest.raises(Geometry19PipelineContextError, match="profile differs"):
        resolve_pipeline_context(
            REPO_ROOT, context, pipeline_profile="v2_2_shadow"
        )


def test_pipeline_stages_do_not_load_physics_yaml_directly() -> None:
    stage_modules = (
        "geometry19_candidates.py",
        "geometry19_physics.py",
        "geometry19_refinement.py",
        "geometry19_nsga2.py",
        "geometry19_delivery.py",
    )
    for filename in stage_modules:
        source = (REPO_ROOT / "src/ai" / filename).read_text(encoding="utf-8")
        assert "ModelConfig.from_yaml" not in source
        assert "ConstraintProfile.from_yaml" not in source
        assert "pipeline_debug_v1" not in source
        assert "wp19_23_inverse_closure_v1.yaml" not in source

    cli_source = (
        REPO_ROOT / "scripts/run_wp19_23_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "--pipeline-profile" in cli_source
    assert "load_wp19_23_contract" not in cli_source


def test_injected_design_store_does_not_reload_physics_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = build_pipeline_context(REPO_ROOT)
    design = _reference_design(context)
    store = Geometry19DesignStore(
        REPO_ROOT,
        tmp_path / "store",
        physics_runtime=context.physics_runtime,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Injected design store attempted to reload physics YAML")

    monkeypatch.setattr(
        "src.visualization.geometry19_service.ModelConfig.from_yaml", forbidden
    )
    monkeypatch.setattr(
        "src.visualization.geometry19_service.ConstraintProfile.from_yaml", forbidden
    )
    payload = store.solve(design)
    assert payload["physics_bundle_hash"] == context.physics_runtime.physics_bundle_hash
    assert payload["constraints"]["constraint_profile"] == (
        context.physics_runtime.constraint_profile.profile_version
    )


def test_runtime_builder_is_the_only_repository_physics_constructor() -> None:
    from src.physics import runtime as runtime_module

    source = inspect.getsource(runtime_module.build_physics_runtime)
    assert "ModelConfig.from_yaml" in source
    assert "ConstraintProfile.from_yaml" in source
