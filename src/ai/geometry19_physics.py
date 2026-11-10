"""True Geometry19 V2.1 evaluation and paired D-Flow acceptance for WP19-23C."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.ai.geometry19_inverse_contract import (
    DiagnosticAcceptance,
    Geometry19InverseTarget,
    load_target,
)
from src.ai.geometry19_hit_alignment import (
    evaluation_hit_key,
    metrics_from_relative_errors,
)
from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_pipeline_context import PipelineContext, resolve_pipeline_context
from src.ai.geometry19_selection import (
    ConstraintDominanceContract,
    assess_constraint_dominance,
    attach_dispositions,
    build_disposition_rows,
    candidate_sort_key,
    constraint_state,
    rank_delivery_candidates,
)
from src.physics import (
    ConstraintProfile,
    EngineResult,
    Environment,
    Geometry19,
    ModelConfig,
    OperatingPointSelectionPolicy,
    assess_constraint_profile,
    solve_design_point,
)
from src.physics.runtime import PhysicsRuntime


class Geometry19PhysicsError(ValueError):
    """Raised when a WP19-23C run violates its true-physics contract."""


def validate_evaluation_budget(
    evaluation_budget: int | None, available_n_keep: int
) -> int:
    """Return the audit count after enforcing the frozen power-of-two rule."""

    if evaluation_budget is None:
        return available_n_keep
    if (
        isinstance(evaluation_budget, bool)
        or evaluation_budget <= 0
        or evaluation_budget & (evaluation_budget - 1)
    ):
        raise Geometry19PhysicsError(
            "evaluation_budget must be a positive power of two"
        )
    if evaluation_budget > available_n_keep:
        raise Geometry19PhysicsError(
            "evaluation_budget cannot exceed the frozen N_keep"
        )
    return evaluation_budget


PhysicsSolver = Callable[..., EngineResult]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n"
            )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Geometry19PhysicsError(f"Cannot read candidate ledger: {path}") from exc


def true_error_metrics(
    acceptance: DiagnosticAcceptance,
    target: Geometry19InverseTarget,
    performance: Mapping[str, Any],
) -> tuple[dict[str, float], float]:
    """Return three non-compensating axis errors and their ranking-only mean."""

    errors = acceptance.normalized_errors(target, performance)
    total = sum(errors.values()) / len(errors)
    return errors, total


def choose_paired_variant(
    direct: Mapping[str, Any],
    dflow: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare variants for offline audit; production never calls this selector."""

    direct_error = (
        float(direct["total_true_error"])
        if direct.get("physics_acceptable") and direct.get("total_true_error") is not None
        else math.inf
    )
    dflow_error = (
        float(dflow["total_true_error"])
        if dflow.get("physics_acceptable") and dflow.get("total_true_error") is not None
        else math.inf
    )
    direct_feasible, direct_violation = constraint_state(direct)
    dflow_feasible, dflow_violation = constraint_state(dflow)
    direct_key = (*evaluation_hit_key(direct), not direct_feasible, direct_violation)
    dflow_key = (*evaluation_hit_key(dflow), not dflow_feasible, dflow_violation)
    if dflow_key < direct_key:
        selected = dflow
        reason = "DFLOW_HIT_ALIGNED_LEXICOGRAPHIC_IMPROVEMENT"
    elif bool(direct.get("physics_acceptable")):
        selected = direct
        reason = (
            "DIRECT_RETAINED_DFLOW_NOT_HIT_ALIGNED_BETTER"
            if bool(dflow.get("physics_acceptable"))
            else "DIRECT_RETAINED_DFLOW_PHYSICS_INVALID"
        )
    else:
        selected = None
        reason = "BOTH_VARIANTS_PHYSICS_INVALID"
    return {
        "candidate_id": int(dflow["candidate_id"]),
        "direct_evaluation_id": direct["evaluation_id"],
        "dflow_evaluation_id": dflow["evaluation_id"],
        "direct_total_true_error": None if not math.isfinite(direct_error) else direct_error,
        "dflow_total_true_error": None if not math.isfinite(dflow_error) else dflow_error,
        "selected_variant": selected["variant"] if selected is not None else None,
        "selected_evaluation_id": selected["evaluation_id"] if selected is not None else None,
        "selected_total_true_error": (
            selected["total_true_error"] if selected is not None else None
        ),
        "direct_hit_alignment": direct.get("hit_alignment"),
        "dflow_hit_alignment": dflow.get("hit_alignment"),
        "selected_hit_alignment": selected.get("hit_alignment") if selected else None,
        "development_diagnostic_hit": bool(
            selected is not None and selected["development_diagnostic_hit"]
        ),
        "acceptance_reason": reason,
    }


def _prefilter_failure_evaluation(
    candidate_id: int,
    variant: str,
    failure_code: str,
    constraint_dominance_contract: ConstraintDominanceContract | None = None,
) -> dict[str, Any]:
    value = {
        "evaluation_id": f"{variant}:{candidate_id}",
        "candidate_id": candidate_id,
        "variant": variant,
        "design_hash": None,
        "geometry19": None,
        "solver_converged": False,
        "solver_failure_code": failure_code,
        "solver_message": "Candidate failed before V2.1 evaluation",
        "solver_evaluations": 0,
        "solver_iterations": 0,
        "flow_root_count": 0,
        "selected_flow_root_index": None,
        "selected_flow_root_mass_flow_air_kg_s": None,
        "root_classification": "PREFILTER_INVALID",
        "performance": None,
        "relative_errors": None,
        "total_true_error": None,
        "total_true_error_metric": "equal_mean_relative_error",
        "hit_alignment": None,
        "constraint_assessment": None,
        "constraint_groups": None,
        "conservation": None,
        "physics_acceptable": False,
        "development_diagnostic_hit": False,
        "formal_hit": None,
    }
    if constraint_dominance_contract is not None:
        value["constraint_dominance"] = assess_constraint_dominance(
            value, constraint_dominance_contract
        )
    return value


def evaluate_geometry19_candidate(
    candidate_id: int,
    variant: str,
    geometry: Mapping[str, Any],
    design_hash: str,
    target: Geometry19InverseTarget,
    acceptance: DiagnosticAcceptance,
    model_config: ModelConfig,
    constraint_profile: ConstraintProfile,
    environment: Environment,
    *,
    solver: PhysicsSolver = solve_design_point,
    physics_runtime: PhysicsRuntime | None = None,
    constraint_dominance_contract: ConstraintDominanceContract | None = None,
) -> dict[str, Any]:
    design = Geometry19.from_mapping(geometry)
    if physics_runtime is None:
        result = solver(
            design,
            environment,
            model_config,
            operating_point_policy=(
                OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1
            ),
        )
    else:
        if (
            model_config.calibration_pack_hash
            != physics_runtime.model_config.calibration_pack_hash
            or constraint_profile.source_hash
            != physics_runtime.constraint_profile.source_hash
        ):
            raise Geometry19PhysicsError(
                "Candidate evaluator dependencies differ from the physics runtime"
            )
        result = physics_runtime.evaluate(design, environment)
    assessment = assess_constraint_profile(
        result,
        constraint_profile,
        conservation_tolerances=(
            physics_runtime.conservation_tolerances
            if physics_runtime is not None
            else None
        ),
    )
    performance = dict(result.performance) if result.performance else None
    errors: dict[str, float] | None = None
    total_error: float | None = None
    if assessment.numerically_valid and performance is not None:
        candidate_errors, candidate_total = true_error_metrics(
            acceptance, target, performance
        )
        if math.isfinite(candidate_total):
            errors = candidate_errors
            total_error = candidate_total
    physics_acceptable = bool(assessment.accepted and total_error is not None)
    diagnostic_hit = bool(
        physics_acceptable
        and acceptance.is_diagnostic_hit(
            target,
            performance or {},
            solver_converged=result.solver.converged,
            numerical_hard_constraints_passed=assessment.numerically_valid,
        )
    )
    hit_alignment = (
        metrics_from_relative_errors(errors, acceptance.relative_tolerances)
        if physics_acceptable and errors is not None
        else None
    )
    if hit_alignment is not None and bool(hit_alignment["strict_hit"]) != diagnostic_hit:
        raise Geometry19PhysicsError("Hit-alignment metric disagrees with diagnostic hit")
    root_count = int(result.operating_point.get("flow_root_count", 0))
    if not result.solver.converged:
        root_classification = "PHYSICS_FAILED"
    elif root_count == 1:
        root_classification = "CONVERGED_UNIQUE"
    else:
        root_classification = "CONVERGED_SELECTED_MULTIPLE"
    failure = result.solver.failure_code
    value = {
        "evaluation_id": f"{variant}:{candidate_id}",
        "candidate_id": candidate_id,
        "variant": variant,
        "design_hash": design_hash,
        "geometry19": design.as_mapping(),
        "solver_converged": result.solver.converged,
        "solver_failure_code": failure.value if failure is not None else None,
        "solver_message": result.solver.message,
        "solver_evaluations": result.solver.evaluations,
        "solver_iterations": result.solver.iterations,
        "flow_root_count": root_count,
        "selected_flow_root_index": result.operating_point.get("selected_flow_root_index"),
        "selected_flow_root_mass_flow_air_kg_s": result.operating_point.get(
            "selected_flow_root_mass_flow_air_kg_s"
        ),
        "root_classification": root_classification,
        "performance": performance,
        "relative_errors": errors,
        "total_true_error": total_error,
        "total_true_error_metric": "equal_mean_relative_error",
        "hit_alignment": hit_alignment,
        "constraint_assessment": assessment.as_mapping(),
        "constraint_groups": {
            "thermodynamic": dict(result.thermodynamic_constraints),
            "assembly": dict(result.assembly_constraints),
            "structural": dict(result.structural_constraints),
            "thermal": dict(result.thermal_constraints),
            "manufacturing": dict(result.manufacturing_constraints),
        },
        "conservation": asdict(result.conservation),
        "physics_acceptable": physics_acceptable,
        "development_diagnostic_hit": diagnostic_hit,
        "formal_hit": None,
        "physics_backend_id": (
            physics_runtime.backend_id if physics_runtime is not None else "legacy_direct_solver"
        ),
        "physics_bundle_hash": (
            physics_runtime.physics_bundle_hash if physics_runtime is not None else None
        ),
        "physics_model_version": result.physics_model_version,
        "calibration_pack_hash": result.calibration_pack_hash,
        "calibration_status": result.calibration_status,
    }
    if constraint_dominance_contract is not None:
        value["constraint_dominance"] = assess_constraint_dominance(
            value, constraint_dominance_contract
        )
    return value


def _candidate_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    values = {int(row["candidate_id"]): row for row in rows}
    if len(values) != len(rows):
        raise Geometry19PhysicsError("Candidate ledger contains duplicate candidate_id")
    return values


def run_true_physics_stage(
    repository_root: str | Path,
    target_path: str | Path,
    *,
    candidate_root: str | Path | None = None,
    output_root: str | Path | None = None,
    sampling_mode: str = "direct",
    evaluation_budget: int | None = None,
    solver: PhysicsSolver | None = None,
    pipeline_context: PipelineContext | None = None,
) -> dict[str, Any]:
    """Evaluate one final direct or D-Flow-assisted candidate set with V2.1.

    ``evaluation_budget`` is an offline-audit control.  Production leaves it
    unset and therefore evaluates every frozen ``N_keep`` candidate.  When it
    is set, it must be a positive power of two no larger than ``N_keep`` so an
    equal-budget direct/D-Flow campaign can compare the same number of V2.1
    calls even when the two filters retain different powers of two.
    """

    root = Path(repository_root).resolve()
    target_source = Path(target_path).resolve()
    target = load_target(target_source)
    context = resolve_pipeline_context(root, pipeline_context, solver=solver)
    contract_path = context.contract_path
    contract = context.contract
    protocol = context.protocol
    physics_runtime = context.physics_runtime
    candidates = Path(candidate_root).resolve() if candidate_root else (
        root
        / protocol["candidate_generation_artifacts"]["runtime_root"]
        / target.target_id
    )
    runtime = Path(output_root).resolve() if output_root else (
        root
        / protocol["artifact_protocol"]["runtime_root"]
        / target.target_id
        / "wp19_23c"
    )
    if runtime.exists():
        raise Geometry19PhysicsError(f"Refusing to overwrite physics run: {runtime}")
    runtime.mkdir(parents=True)
    try:
        if sampling_mode not in {"direct", "dflow"}:
            raise Geometry19PhysicsError("sampling_mode must be 'direct' or 'dflow'")
        candidate_summary_path = candidates / sampling_mode / "summary.json"
        candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
        current_protocol_hash = context.provenance_hashes["protocol_hash"]
        if (
            candidate_summary["target_id"] != target.target_id
            or candidate_summary["sampling_mode"] != sampling_mode
        ):
            raise Geometry19PhysicsError(
                f"{sampling_mode} candidate summary identity mismatch"
            )
        if candidate_summary["hashes"]["protocol_hash"] != current_protocol_hash:
            raise Geometry19PhysicsError(f"{sampling_mode} candidate protocol hash is stale")

        candidate_rows = _read_jsonl(
            candidates / sampling_mode / "candidate_filter_ledger.jsonl"
        )
        candidates_by_id = _candidate_by_id(candidate_rows)
        model_config = physics_runtime.model_config
        constraint_profile = physics_runtime.constraint_profile
        dominance_contract = context.constraint_dominance
        schema_path = physics_runtime.source_paths["design_schema"]
        schema = physics_runtime.design_schema
        fields = list(schema["design_variables"])
        expected_calibration_hash = candidate_summary["hashes"]["calibration_hash"]
        if model_config.calibration_pack_hash != expected_calibration_hash:
            raise Geometry19PhysicsError("Candidate and V2.1 calibration hashes differ")
        candidate_physics_bundle_hash = candidate_summary["hashes"].get(
            "physics_bundle_hash"
        )
        if (
            candidate_physics_bundle_hash is not None
            and candidate_physics_bundle_hash != physics_runtime.physics_bundle_hash
        ):
            raise Geometry19PhysicsError("Candidate and physics runtime bundles differ")
        environment = Environment()
        evaluations: list[dict[str, Any]] = []
        cache: dict[str, dict[str, Any]] = {}

        def evaluate(row: Mapping[str, Any], variant: str) -> dict[str, Any]:
            candidate_id = int(row["candidate_id"])
            if not row.get("valid") or not row.get("geometry19") or not row.get("design_hash"):
                value = _prefilter_failure_evaluation(
                    candidate_id,
                    variant,
                    str(row.get("failure_code") or "PREFILTER_INVALID"),
                    dominance_contract,
                )
                evaluations.append(value)
                return value
            key = str(row["design_hash"])
            if key not in cache:
                cache[key] = evaluate_geometry19_candidate(
                    candidate_id,
                    variant,
                    row["geometry19"],
                    str(row["design_hash"]),
                    target,
                    contract.diagnostic_acceptance,
                    model_config,
                    constraint_profile,
                    environment,
                    physics_runtime=physics_runtime,
                    constraint_dominance_contract=dominance_contract,
                )
            value = dict(cache[key])
            value["candidate_id"] = candidate_id
            value["evaluation_id"] = f"{variant}:{candidate_id}"
            value["variant"] = variant
            evaluations.append(value)
            return value

        all_kept_candidate_ids = candidate_summary["candidate_protocol"][
            "kept_candidate_ids"
        ]
        evaluated_count = validate_evaluation_budget(
            evaluation_budget, len(all_kept_candidate_ids)
        )
        kept_candidate_ids = all_kept_candidate_ids[:evaluated_count]
        selected_pool = [
            evaluate(candidates_by_id[int(candidate_id)], sampling_mode)
            for candidate_id in kept_candidate_ids
        ]
        hits = [row for row in selected_pool if row["development_diagnostic_hit"]]
        valid = [row for row in selected_pool if row["physics_acceptable"]]
        valid.sort(key=candidate_sort_key)
        stop = bool(hits)
        delivered = (
            rank_delivery_candidates(
                hits, int(protocol["delivery"]["maximum_candidates"]), fields
            )
            if stop
            else []
        )
        dispositions = build_disposition_rows(
            evaluations,
            delivered,
            stage="WP19-23C",
            next_stage=None if stop else "WP19-23D",
            contract=dominance_contract,
        )
        attach_dispositions(evaluations, dispositions)
        attach_dispositions(delivered, dispositions)
        summary = {
            "work_package": "WP19-23C",
            "status": "DEVELOPMENT_DIAGNOSTIC_HIT" if stop else "NO_GENERATION_STAGE_HIT",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target.target_id,
            "sampling_mode": sampling_mode,
            "evaluation_budget": {
                "mode": "offline_equal_budget_cap"
                if evaluation_budget is not None
                else "production_all_n_keep",
                "requested": evaluation_budget,
                "available_n_keep": len(all_kept_candidate_ids),
                "evaluated": len(kept_candidate_ids),
            },
            "formal_use_allowed": False,
            "formal_hit": None,
            "downstream_stop": stop,
            "next_stage": "DELIVERY" if stop else "WP19-23D_REFINEMENT",
            "counts": {
                "candidate_ids_kept": len(kept_candidate_ids),
                "selected_candidate_count": len(selected_pool),
                "physics_acceptable": sum(
                    bool(row["physics_acceptable"]) for row in selected_pool
                ),
                "diagnostic_hits": sum(
                    bool(row["development_diagnostic_hit"]) for row in selected_pool
                ),
                "unique_v2_1_evaluations": len(cache),
                "unique_physics_evaluations": len(cache),
                "physics_ledger_rows": len(evaluations),
                "delivered_count": len(delivered),
                "constraint_feasible": sum(
                    bool((row.get("constraint_dominance") or {}).get("constraint_feasible"))
                    for row in selected_pool
                ),
            },
            "best_approximation": valid[0] if valid else None,
            "delivered_evaluation_ids": [row["evaluation_id"] for row in delivered],
            "disposition_counts": {
                reason: sum(row["disposition_reason"] == reason for row in dispositions)
                for reason in sorted({row["disposition_reason"] for row in dispositions})
            },
            "hashes": {
                "protocol_hash": current_protocol_hash,
                "target_hash": sha256_file(target_source),
                "calibration_hash": model_config.calibration_pack_hash,
                "physics_backend_id": physics_runtime.backend_id,
                "physics_bundle_hash": physics_runtime.physics_bundle_hash,
                "constraint_profile_hash": constraint_profile.source_hash,
                "constraint_dominance_hash": dominance_contract.source_hash,
                "operating_point_policy_hash": sha256_file(
                    root / protocol["contracts"]["operating_point_policy"]
                ),
                "candidate_summary_hash": sha256_file(candidate_summary_path),
            },
        }
        _write_json(runtime / "target.json", target.as_mapping())
        _write_jsonl(runtime / "physics_forward_ledger.jsonl", evaluations)
        _write_jsonl(runtime / "candidate_disposition_ledger.jsonl", dispositions)
        _write_json(runtime / "delivered_candidates.json", delivered)
        _write_json(runtime / "summary.json", summary)
        _write_json(
            runtime / "manifest.json",
            {
                "protocol_version": protocol["protocol_version"],
                "target_id": target.target_id,
                "stage": "WP19-23C",
                "required_files": [
                    "manifest.json",
                    "target.json",
                    "physics_forward_ledger.jsonl",
                    "candidate_disposition_ledger.jsonl",
                    "delivered_candidates.json",
                    "summary.json",
                ],
                "summary_sha256": sha256_file(runtime / "summary.json"),
            },
        )
        return summary
    except Exception as exc:
        _write_json(
            runtime / "failure.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
