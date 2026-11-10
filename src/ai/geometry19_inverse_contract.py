"""Frozen configuration and target contracts for WP19-23."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.ai.geometry19_hit_alignment import (
    Geometry19HitAlignmentError,
    validate_hit_alignment_config,
)


TARGET_CONTRACT_VERSION = "wp19_23_target_v1"
DIAGNOSTIC_PROFILE = "wp19_23_development_diagnostic_v1"
TARGET_TYPES = {
    "ID",
    "INTERPOLATION",
    "BOUNDARY_SPARSE",
    "OUT_OF_EMPIRICAL_SUPPORT",
}
EXPECTED_STATUSES = {"KNOWN_FEASIBLE", "UNKNOWN", "EXPECTED_INFEASIBLE"}
TARGET_FIELDS = {
    "target_contract_version",
    "target_id",
    "target_type",
    "expected_status",
    "thrust_N",
    "tsfc_kg_N_s",
    "mass_kg",
    "acceptance_profile",
}
OUTPUT_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
FROZEN_RAW_CANDIDATE_COUNT = 256
GENERATOR_VARIANT_NAMES = ("baseline", "bounded-sigmoid")
FORWARD_MODEL_TAGS = ("performance3_log", "s0")


class Geometry19InverseContractError(ValueError):
    """Raised when a WP19-23 protocol or target violates its frozen contract."""


def _canonical_target_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "acceptance_profile": str(value["acceptance_profile"]),
        "expected_status": str(value["expected_status"]),
        "mass_kg": float(value["mass_kg"]),
        "target_contract_version": str(value["target_contract_version"]),
        "target_type": str(value["target_type"]),
        "thrust_N": float(value["thrust_N"]),
        "tsfc_kg_N_s": float(value["tsfc_kg_N_s"]),
    }


def derive_target_id(value: Mapping[str, Any]) -> str:
    """Return the frozen SHA256 identifier for a target mapping."""

    encoded = json.dumps(
        _canonical_target_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Geometry19InverseTarget:
    target_contract_version: str
    target_id: str
    target_type: str
    expected_status: str
    thrust_N: float
    tsfc_kg_N_s: float
    mass_kg: float
    acceptance_profile: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Geometry19InverseTarget":
        if not isinstance(value, Mapping):
            raise Geometry19InverseContractError("Target document must be a mapping")
        keys = set(value)
        missing = TARGET_FIELDS - keys
        extra = keys - TARGET_FIELDS
        if missing or extra:
            raise Geometry19InverseContractError(
                f"Target fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if value["target_contract_version"] != TARGET_CONTRACT_VERSION:
            raise Geometry19InverseContractError("Target contract version mismatch")
        if value["acceptance_profile"] != DIAGNOSTIC_PROFILE:
            raise Geometry19InverseContractError("Target acceptance profile mismatch")
        if value["target_type"] not in TARGET_TYPES:
            raise Geometry19InverseContractError("Unsupported target_type")
        if value["expected_status"] not in EXPECTED_STATUSES:
            raise Geometry19InverseContractError("Unsupported expected_status")
        outputs = {name: float(value[name]) for name in OUTPUT_NAMES}
        if any(not math.isfinite(number) or number <= 0.0 for number in outputs.values()):
            raise Geometry19InverseContractError(
                "Target thrust, TSFC, and mass must be finite and positive"
            )
        expected_id = derive_target_id(value)
        if value["target_id"] != expected_id:
            raise Geometry19InverseContractError("Target ID does not match canonical content")
        return cls(
            target_contract_version=TARGET_CONTRACT_VERSION,
            target_id=str(value["target_id"]),
            target_type=str(value["target_type"]),
            expected_status=str(value["expected_status"]),
            thrust_N=outputs["thrust_N"],
            tsfc_kg_N_s=outputs["tsfc_kg_N_s"],
            mass_kg=outputs["mass_kg"],
            acceptance_profile=DIAGNOSTIC_PROFILE,
        )

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticAcceptance:
    relative_tolerances: dict[str, float]
    epsilons: dict[str, float]
    formal_hit_rate_ready: bool

    def normalized_errors(
        self,
        target: Geometry19InverseTarget,
        actual: Mapping[str, Any],
    ) -> dict[str, float]:
        target_values = target.as_mapping()
        errors: dict[str, float] = {}
        for name in OUTPUT_NAMES:
            actual_value = float(actual[name])
            if not math.isfinite(actual_value):
                errors[name] = math.inf
                continue
            denominator = max(abs(float(target_values[name])), self.epsilons[name])
            errors[name] = abs(actual_value - float(target_values[name])) / denominator
        return errors

    def is_diagnostic_hit(
        self,
        target: Geometry19InverseTarget,
        actual: Mapping[str, Any],
        *,
        solver_converged: bool,
        numerical_hard_constraints_passed: bool,
    ) -> bool:
        if not solver_converged or not numerical_hard_constraints_passed:
            return False
        errors = self.normalized_errors(target, actual)
        return all(errors[name] <= self.relative_tolerances[name] for name in OUTPUT_NAMES)


@dataclass(frozen=True)
class WP1923Contract:
    source_path: Path
    payload: dict[str, Any]
    diagnostic_acceptance: DiagnosticAcceptance

    @property
    def required_run_artifacts(self) -> tuple[str, ...]:
        return tuple(self.payload["artifact_protocol"]["required_run_artifacts"])

    @property
    def default_generator_variant(self) -> str:
        return str(self.payload["models"]["generator"]["default_variant"])

    def generator_variant(self, name: str | None = None) -> dict[str, Any]:
        resolved = self.default_generator_variant if name is None else str(name)
        try:
            return dict(self.payload["models"]["generator"]["variants"][resolved])
        except KeyError as exc:
            raise Geometry19InverseContractError(
                f"Unsupported generator variant: {resolved}"
            ) from exc

    @property
    def default_forward_model_tag(self) -> str:
        return str(self.payload["models"]["forward"]["default_tag"])

    def forward_model(self, tag: str | None = None) -> dict[str, Any]:
        resolved = self.default_forward_model_tag if tag is None else str(tag)
        try:
            return dict(self.payload["models"]["forward"]["variants"][resolved])
        except KeyError as exc:
            raise Geometry19InverseContractError(
                f"Unsupported forward model tag: {resolved}"
            ) from exc


def load_target(path: str | Path) -> Geometry19InverseTarget:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Geometry19InverseContractError(f"Cannot load target: {source}") from exc
    return Geometry19InverseTarget.from_mapping(payload)


def _deep_merge_protocol(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_protocol(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_protocol_payload(
    source: Path, seen: frozenset[Path] = frozenset()
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    resolved = source.resolve()
    if resolved in seen:
        raise Geometry19InverseContractError("Protocol overlay inheritance cycle")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Geometry19InverseContractError(f"Cannot load protocol: {resolved}") from exc
    if not isinstance(payload, dict):
        raise Geometry19InverseContractError("Protocol document must be a mapping")
    if "extends" not in payload:
        return payload, (resolved,)
    if set(payload) != {"overlay_version", "extends", "overrides"}:
        raise Geometry19InverseContractError("Protocol overlay keys mismatch")
    if payload["overlay_version"] != "geometry19_pipeline_protocol_overlay_v1":
        raise Geometry19InverseContractError("Protocol overlay version mismatch")
    extends = payload["extends"]
    overrides = payload["overrides"]
    if not isinstance(extends, str) or not isinstance(overrides, dict):
        raise Geometry19InverseContractError("Protocol overlay extends/overrides are invalid")
    repository_root = Path(__file__).resolve().parents[2]
    base_path = (repository_root / extends).resolve()
    try:
        base_path.relative_to(repository_root)
    except ValueError as exc:
        raise Geometry19InverseContractError("Protocol overlay escapes repository root") from exc
    allowed_overrides = {
        "status",
        "contracts",
        "generation",
        "dflow",
        "physics_evaluation",
        "refinement",
        "optimizer",
        "delivery",
        "artifact_protocol",
        "shadow",
    }
    if not set(overrides) <= allowed_overrides:
        raise Geometry19InverseContractError(
            f"Protocol overlay contains forbidden sections: "
            f"{sorted(set(overrides) - allowed_overrides)}"
        )
    base, dependencies = _load_protocol_payload(
        base_path, seen | {resolved}
    )
    return _deep_merge_protocol(base, overrides), (*dependencies, resolved)


def load_wp19_23_contract(path: str | Path) -> WP1923Contract:
    source = Path(path)
    payload, protocol_dependencies = _load_protocol_payload(source)
    required_sections = {
        "protocol_version",
        "work_package",
        "formal_use_allowed",
        "contracts",
        "target_outputs",
        "diagnostic_acceptance",
        "hit_alignment",
        "models",
        "generation",
        "dflow",
        "candidate_filter",
        "physics_evaluation",
        "refinement",
        "optimizer",
        "delivery",
        "artifact_protocol",
    }
    missing = required_sections - set(payload)
    if missing:
        raise Geometry19InverseContractError(
            f"Protocol is missing sections: {sorted(missing)}"
        )
    payload["_protocol_dependencies"] = [
        str(item) for item in protocol_dependencies
    ]
    if payload["protocol_version"] != "wp19_23_inverse_closure_v1":
        raise Geometry19InverseContractError("Protocol version mismatch")
    if payload["work_package"] != "WP19-23":
        raise Geometry19InverseContractError("Work package mismatch")
    if payload["formal_use_allowed"] is not False:
        raise Geometry19InverseContractError("WP19-23 v1 must remain development-only")
    if payload["target_outputs"]["ordered_names"] != list(OUTPUT_NAMES):
        raise Geometry19InverseContractError("Target output order mismatch")
    try:
        validate_hit_alignment_config(
            payload["hit_alignment"],
            payload["diagnostic_acceptance"]["relative_tolerances"],
        )
    except Geometry19HitAlignmentError as exc:
        raise Geometry19InverseContractError(str(exc)) from exc

    generator = payload["models"]["generator"]
    variants = generator.get("variants")
    if not isinstance(variants, dict) or tuple(variants) != GENERATOR_VARIANT_NAMES:
        raise Geometry19InverseContractError("Generator variant contract mismatch")
    if generator.get("default_variant") != "bounded-sigmoid":
        raise Geometry19InverseContractError(
            "Bounded-sigmoid must remain the default generator variant"
        )
    expected_variants = {
        "baseline": {
            "model_tag": "novaturbo_geometry19_gpu_sinkhorn_cfm4_film_residual_v1",
            "checkpoint": (
                "data/generator_models/geometry19/"
                "wp19_23_cfm4_gpu_sinkhorn_seed43_v1/baseline/best.pt"
            ),
            "normalizer": (
                "data/generator_models/geometry19/"
                "wp19_23_cfm4_gpu_sinkhorn_seed43_v1/baseline/normalizer.json"
            ),
            "network_architecture_modified": False,
        },
        "bounded-sigmoid": {
            "model_tag": (
                "novaturbo_geometry19_gpu_sinkhorn_cfm4_film_residual_bounded_sigmoid_v1"
            ),
            "checkpoint": (
                "data/generator_models/geometry19/"
                "wp19_23_cfm4_bounded_sigmoid_seed43_v1/baseline/best.pt"
            ),
            "normalizer": (
                "data/generator_models/geometry19/"
                "wp19_23_cfm4_bounded_sigmoid_seed43_v1/baseline/normalizer.json"
            ),
            "network_architecture_modified": True,
        },
    }
    for name in GENERATOR_VARIANT_NAMES:
        variant = variants[name]
        expected = expected_variants[name]
        if (
            any(variant.get(key) != value for key, value in expected.items())
            or int(variant.get("training_seed", -1)) != 43
        ):
            raise Geometry19InverseContractError(
                f"Generator variant metadata mismatch: {name}"
            )

    forward = payload["models"]["forward"]
    forward_variants = forward.get("variants")
    if not isinstance(forward_variants, dict) or tuple(forward_variants) != FORWARD_MODEL_TAGS:
        raise Geometry19InverseContractError("Forward model tag contract mismatch")
    if forward.get("default_tag") not in forward_variants:
        raise Geometry19InverseContractError(
            "Default forward model tag must name a configured variant"
        )
    expected_forward = {
        "performance3_log": {
            "model_tag": "novaturbo_geometry19_performance3_log_seed43_v1",
            "weights": (
                "data/inverse_design/wp19_22_performance_surrogate_v2_ablation/"
                "formal/models/performance3_log/seed_43"
            ),
            "output_transform": "natural_log",
            "output_contract": "performance3_only",
            "use_constraint_loss": False,
            "use_feasibility_loss": False,
        },
        "s0": {
            "model_tag": "novaturbo_geometry19_forward_gate_k_s0_legacy_mse_seed43_v1",
            "weights": (
                "data/inverse_design/wp19_22_gate_k_hit_aligned_surrogate_v1/"
                "models/s0_legacy_mse/seed_43"
            ),
            "output_transform": "identity",
            "output_contract": "performance3_plus_constraint_residuals",
            "use_constraint_loss": True,
            "use_feasibility_loss": True,
        },
    }
    for tag in FORWARD_MODEL_TAGS:
        if any(
            forward_variants[tag].get(key) != value
            for key, value in expected_forward[tag].items()
        ):
            raise Geometry19InverseContractError(
                f"Forward model metadata mismatch: {tag}"
            )

    generation = payload["generation"]
    if generation["raw_candidate_count"] != FROZEN_RAW_CANDIDATE_COUNT:
        raise Geometry19InverseContractError("WP19-23 requires N_raw=256")
    if generation["generator_call_count_per_target"] != 1:
        raise Geometry19InverseContractError("Generator must be called once per target")
    if generation["refill_allowed"] or generation["candidate_copy_allowed"]:
        raise Geometry19InverseContractError("Refill and candidate copying must be disabled")
    if generation["default_sampling_mode"] not in generation["supported_sampling_modes"]:
        raise Geometry19InverseContractError("Default sampling mode is unsupported")

    candidate_filter = payload["candidate_filter"]
    if candidate_filter["trim_rule"] != "largest_power_of_two_not_above_valid_count":
        raise Geometry19InverseContractError("Candidate trim rule mismatch")
    if (
        candidate_filter["surrogate_reranking_before_trim_allowed"] is not True
        or candidate_filter["select_before_physics"]
        != "predicted_rho_ascending_then_candidate_id"
    ):
        raise Geometry19InverseContractError("Hit-aligned surrogate reranking is required")
    if (
        candidate_filter["discrete_policy"]
        != "unconditional_nearest_integer"
        or candidate_filter["discrete_tolerance_status"]
        != "disabled_unconditional_projection_v2"
        or candidate_filter["discrete_absolute_tolerance_count"] is not None
        or candidate_filter["discrete_tolerance_boundary_inclusive"] is not True
        or candidate_filter["discrete_tie_break"] != "numpy_rint_nearest_even"
    ):
        raise Geometry19InverseContractError("Unconditional discrete projection contract mismatch")

    optimizer = payload["optimizer"]
    if optimizer["constraint_dominance"] not in {
        "geometry19_constraint_dominance_v1",
        "geometry19_constraint_dominance_v2_2_c4_v1",
    }:
        raise Geometry19InverseContractError("NSGA-II constraint dominance mismatch")
    if (
        optimizer.get("selection_order")
        != "failed_axes_then_rho_then_sum_then_constraint_violation"
    ):
        raise Geometry19InverseContractError("NSGA-II hit-aligned selection mismatch")
    minimum_population = int(optimizer["minimum_population_size"])
    maximum_population = int(optimizer["maximum_population_size"])
    if (
        minimum_population <= 0
        or maximum_population < minimum_population
        or minimum_population & (minimum_population - 1)
        or maximum_population & (maximum_population - 1)
    ):
        raise Geometry19InverseContractError(
            "NSGA-II population bounds must be ordered powers of two"
        )
    if int(optimizer["maximum_physics_evaluations"]) < maximum_population * 2:
        raise Geometry19InverseContractError(
            "NSGA-II budget must fund at least one maximum-size generation"
        )

    diagnostic = payload["diagnostic_acceptance"]
    if diagnostic["formal_hit_rate_ready"] is not False:
        raise Geometry19InverseContractError("Formal hit rate must remain unavailable")
    if diagnostic["formal_policy_override_allowed"] is not False:
        raise Geometry19InverseContractError("Diagnostic thresholds cannot override formal policy")
    tolerances = {name: float(diagnostic["relative_tolerances"][name]) for name in OUTPUT_NAMES}
    epsilons = {name: float(diagnostic["epsilons"][name]) for name in OUTPUT_NAMES}
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances.values()):
        raise Geometry19InverseContractError("Diagnostic tolerances must be finite and positive")
    if any(not math.isfinite(value) or value <= 0.0 for value in epsilons.values()):
        raise Geometry19InverseContractError("Diagnostic epsilons must be finite and positive")

    required_artifacts = payload["artifact_protocol"]["required_run_artifacts"]
    if len(required_artifacts) != len(set(required_artifacts)):
        raise Geometry19InverseContractError("Required run artifacts must be unique")
    return WP1923Contract(
        source_path=source.resolve(),
        payload=payload,
        diagnostic_acceptance=DiagnosticAcceptance(
            relative_tolerances=tolerances,
            epsilons=epsilons,
            formal_hit_rate_ready=False,
        ),
    )
