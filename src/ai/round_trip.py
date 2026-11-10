"""Physics-in-the-loop round-trip evaluation for conditional generators.

The evaluator deliberately scores ``f(x_generated)`` against the requested
condition.  It never compares a generated design with a single reference
design, because the inverse Brayton problem is one-to-many.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from .baseline import sha256_file
from .dataset import evaluate_design
from .generator_dataset import (
    GeneratorNormalizer,
    condition_names,
    design_names,
    engine_inputs_from_cycle_row,
)
from ..physics.brayton import FlightConditions, solve_brayton_cycle


OK = "OK"
NON_FINITE = "NON_FINITE"
OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
DUPLICATE = "DUPLICATE"
BRAYTON_EXCEPTION = "BRAYTON_EXCEPTION"
INVALID_CYCLE = "INVALID_CYCLE"
NEGATIVE_OR_ZERO_THRUST = "NEGATIVE_OR_ZERO_THRUST"
NON_FINITE_TSFC = "NON_FINITE_TSFC"
NON_FINITE_MASS = "NON_FINITE_MASS"
TARGET_OUT_OF_DOMAIN = "TARGET_OUT_OF_DOMAIN"
SAMPLING_BUDGET_EXHAUSTED = "SAMPLING_BUDGET_EXHAUSTED"
NO_CANDIDATE_WITHIN_TOLERANCE = "NO_CANDIDATE_WITHIN_TOLERANCE"

FAILURE_CODES = (
    NON_FINITE,
    OUT_OF_BOUNDS,
    DUPLICATE,
    BRAYTON_EXCEPTION,
    INVALID_CYCLE,
    NEGATIVE_OR_ZERO_THRUST,
    NON_FINITE_TSFC,
    NON_FINITE_MASS,
    TARGET_OUT_OF_DOMAIN,
    SAMPLING_BUDGET_EXHAUSTED,
    NO_CANDIDATE_WITHIN_TOLERANCE,
)


@dataclass(frozen=True)
class DerivedRoundTripMetrics:
    """Frozen numerical rules used by one WP4 evaluation run."""

    metrics_version: str
    thrust_relative_tolerance: float
    tsfc_relative_tolerance: float
    total_error_tolerance: float
    thrust_epsilon: float
    tsfc_epsilon: float
    thrust_weight: float
    tsfc_weight: float
    violation_weight: float
    normalized_condition_min: Tuple[float, ...]
    normalized_condition_max: Tuple[float, ...]
    domain_margin_fraction: float
    derivation: Dict[str, Any]
    mass_relative_tolerance: float = math.inf
    mass_epsilon: float = 1.0e-15
    mass_weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics_version": self.metrics_version,
            "success_tolerances": {
                "thrust_relative": self.thrust_relative_tolerance,
                "tsfc_relative": self.tsfc_relative_tolerance,
                "mass_relative": self.mass_relative_tolerance,
                "weighted_total": self.total_error_tolerance,
            },
            "epsilons": {
                "thrust_N": self.thrust_epsilon,
                "tsfc_kg_N_s": self.tsfc_epsilon,
                "total_mass_kg": self.mass_epsilon,
            },
            "weights": {
                "thrust": self.thrust_weight,
                "tsfc": self.tsfc_weight,
                "mass": self.mass_weight,
                "constraint_violation": self.violation_weight,
            },
            "normalized_condition_domain": {
                "minimum": list(self.normalized_condition_min),
                "maximum": list(self.normalized_condition_max),
                "margin_fraction": self.domain_margin_fraction,
            },
            "derivation": self.derivation,
        }


@dataclass
class RoundTripResult:
    checked_candidates: pd.DataFrame
    forward_results: pd.DataFrame
    ranked_candidates: pd.DataFrame
    failures: pd.DataFrame
    metrics: Dict[str, Any]


def load_metrics_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "metrics_version",
        "threshold_derivation",
        "epsilons",
        "weights",
        "candidate_validation",
        "selection",
        "reporting",
        "target_domain",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP4 metrics config is missing keys: {sorted(missing)}")
    top_k = [int(value) for value in config["reporting"]["top_k"]]
    if not top_k or any(value <= 0 for value in top_k):
        raise ValueError("WP4 reporting.top_k must contain positive integers")
    if int(config["selection"]["selected_count"]) <= 0:
        raise ValueError("WP4 selection.selected_count must be positive")
    return config


def derive_round_trip_metrics(
    dataset_csv: Path,
    normalizer: GeneratorNormalizer,
    config: Mapping[str, Any],
) -> DerivedRoundTripMetrics:
    """Derive success tolerances and condition domain only from frozen WP2 data."""

    frame = pd.read_csv(dataset_csv)
    names = normalizer.condition_names
    derivation = config["threshold_derivation"]
    reference_split = str(derivation["reference_split"])
    candidate_split = str(derivation["candidate_split"])
    reference = frame.loc[frame["split"] == reference_split].reset_index(drop=True)
    candidates = frame.loc[frame["split"] == candidate_split].reset_index(drop=True)
    if reference.empty or candidates.empty:
        raise ValueError(
            "Cannot derive WP4 thresholds from empty splits: "
            f"reference={reference_split}:{len(reference)}, "
            f"candidate={candidate_split}:{len(candidates)}"
        )

    reference_conditions = reference[names].to_numpy(dtype=float)
    candidate_conditions = candidates[names].to_numpy(dtype=float)
    normalized_reference = normalizer.transform_conditions(reference_conditions)
    normalized_candidates = normalizer.transform_conditions(candidate_conditions)
    _, nearest_indices = cKDTree(normalized_candidates).query(
        normalized_reference, k=1, workers=1
    )
    nearest = candidate_conditions[np.asarray(nearest_indices, dtype=int)]
    eps_f = float(config["epsilons"]["thrust_N"])
    eps_t = float(config["epsilons"]["tsfc_kg_N_s"])
    thrust_errors = np.abs(nearest[:, 0] - reference_conditions[:, 0]) / np.maximum(
        np.abs(reference_conditions[:, 0]), eps_f
    )
    tsfc_errors = np.abs(nearest[:, 1] - reference_conditions[:, 1]) / np.maximum(
        np.abs(reference_conditions[:, 1]), eps_t
    )
    has_mass = "total_mass_kg" in names
    eps_m = float(config["epsilons"].get("total_mass_kg", 1.0e-15))
    mass_errors = (
        np.abs(nearest[:, names.index("total_mass_kg")] - reference_conditions[:, names.index("total_mass_kg")])
        / np.maximum(
            np.abs(reference_conditions[:, names.index("total_mass_kg")]), eps_m
        )
        if has_mass
        else np.asarray([], dtype=float)
    )
    quantile = float(derivation["quantile"])
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("threshold_derivation.quantile must be between zero and one")
    thrust_tolerance = max(
        float(np.quantile(thrust_errors, quantile)),
        float(derivation["minimum_thrust_relative_tolerance"]),
    )
    tsfc_tolerance = max(
        float(np.quantile(tsfc_errors, quantile)),
        float(derivation["minimum_tsfc_relative_tolerance"]),
    )
    mass_tolerance = (
        max(
            float(np.quantile(mass_errors, quantile)),
            float(derivation.get("minimum_mass_relative_tolerance", 0.001)),
        )
        if has_mass
        else math.inf
    )
    weights = config["weights"]
    thrust_weight = float(weights["thrust"])
    tsfc_weight = float(weights["tsfc"])
    mass_weight = float(weights.get("mass", 0.0)) if has_mass else 0.0
    violation_weight = float(weights["constraint_violation"])
    total_tolerance = (
        thrust_weight * thrust_tolerance
        + tsfc_weight * tsfc_tolerance
        + (mass_weight * mass_tolerance if has_mass else 0.0)
    )
    train_min = np.min(normalized_candidates, axis=0)
    train_max = np.max(normalized_candidates, axis=0)
    margin = float(config["target_domain"]["normalized_range_margin_fraction"])

    return DerivedRoundTripMetrics(
        metrics_version=str(config["metrics_version"]),
        thrust_relative_tolerance=thrust_tolerance,
        tsfc_relative_tolerance=tsfc_tolerance,
        total_error_tolerance=total_tolerance,
        thrust_epsilon=eps_f,
        tsfc_epsilon=eps_t,
        thrust_weight=thrust_weight,
        tsfc_weight=tsfc_weight,
        violation_weight=violation_weight,
        normalized_condition_min=tuple(float(value) for value in train_min),
        normalized_condition_max=tuple(float(value) for value in train_max),
        domain_margin_fraction=margin,
        derivation={
            "method": "nearest_train_condition_for_each_test_target",
            "dataset_csv": str(dataset_csv),
            "dataset_sha256": sha256_file(Path(dataset_csv)),
            "reference_split": reference_split,
            "reference_count": len(reference),
            "candidate_split": candidate_split,
            "candidate_count": len(candidates),
            "nearest_neighbour_error_quantile": quantile,
            "raw_quantiles": {
                "thrust_relative": float(np.quantile(thrust_errors, quantile)),
                "tsfc_relative": float(np.quantile(tsfc_errors, quantile)),
                "mass_relative": (
                    float(np.quantile(mass_errors, quantile))
                    if has_mass
                    else math.nan
                ),
            },
            "minimum_tolerance_floors": {
                "thrust_relative": float(
                    derivation["minimum_thrust_relative_tolerance"]
                ),
                "tsfc_relative": float(
                    derivation["minimum_tsfc_relative_tolerance"]
                ),
                "mass_relative": float(
                    derivation.get("minimum_mass_relative_tolerance", 0.001)
                ),
            },
        },
        mass_relative_tolerance=mass_tolerance,
        mass_epsilon=eps_m,
        mass_weight=mass_weight,
    )


class RoundTripEvaluator:
    """Validate, forward-solve, rank, and diversify generated candidates."""

    def __init__(
        self,
        schema: Mapping[str, Any],
        normalizer: GeneratorNormalizer,
        metrics_config: Mapping[str, Any],
        derived_metrics: DerivedRoundTripMetrics,
        forward_solver: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    ) -> None:
        self.schema = dict(schema)
        self.normalizer = normalizer
        self.config = dict(metrics_config)
        self.derived = derived_metrics
        self.design_names = design_names(schema)
        self.condition_names = condition_names(schema)
        self.forward_solver = forward_solver or self._default_forward_solver
        if self.design_names != normalizer.design_names:
            raise ValueError("Schema and normalizer design field order differ")
        if self.condition_names != normalizer.condition_names:
            raise ValueError("Schema and normalizer condition field order differ")

    def _default_forward_solver(self, row: Mapping[str, Any]) -> Any:
        if "total_mass_kg" in self.condition_names:
            return evaluate_design(pd.Series(row))
        return solve_brayton_cycle(
            FlightConditions(), engine_inputs_from_cycle_row(row, self.schema)
        )

    def normalize_target(self, target: Mapping[str, Any]) -> np.ndarray:
        physical = np.asarray(
            [target[name] for name in self.condition_names], dtype=float
        )
        if not np.isfinite(physical).all() or np.any(physical <= 0.0):
            raise ValueError(TARGET_OUT_OF_DOMAIN)
        normalized = np.asarray(
            self.normalizer.transform_conditions(physical), dtype=float
        )
        if not np.isfinite(normalized).all():
            raise ValueError(TARGET_OUT_OF_DOMAIN)
        return normalized

    def target_in_training_domain(self, target: Mapping[str, Any]) -> bool:
        try:
            normalized = self.normalize_target(target)
        except (ValueError, FloatingPointError):
            return False
        lower = np.asarray(self.derived.normalized_condition_min)
        upper = np.asarray(self.derived.normalized_condition_max)
        margin = self.derived.domain_margin_fraction * (upper - lower)
        return bool(np.all(normalized >= lower - margin) and np.all(normalized <= upper + margin))

    def _validate_candidate_arrays(
        self,
        normalized: np.ndarray,
        physical: np.ndarray,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        normalized = np.asarray(normalized, dtype=float)
        physical = np.asarray(physical, dtype=float)
        if normalized.ndim != 2 or normalized.shape[1] != len(self.design_names):
            raise ValueError(
                "normalized candidates must have shape "
                f"[n,{len(self.design_names)}], got {normalized.shape}"
            )
        if physical.shape != normalized.shape:
            raise ValueError(
                f"physical candidates shape {physical.shape} does not match {normalized.shape}"
            )
        if candidate_ids is None:
            ids = np.arange(len(normalized), dtype=int)
        else:
            ids = np.asarray(candidate_ids)
            if ids.ndim != 1 or len(ids) != len(normalized):
                raise ValueError("candidate_ids must match the candidate row count")
            if len(np.unique(ids)) != len(ids):
                raise ValueError("candidate_ids must be unique")
            ids = ids.astype(int)
        validation = self.config["candidate_validation"]
        norm_tolerance = float(validation["normalized_bound_tolerance"])
        physical_tolerance = float(validation["physical_bound_tolerance"])
        duplicate_tolerance = float(validation["duplicate_linf_tolerance"])
        lower = np.asarray(
            [float(field["bounds"][0]) for field in self.schema["design_variables"]]
        )
        upper = np.asarray(
            [float(field["bounds"][1]) for field in self.schema["design_variables"]]
        )
        rows: List[Dict[str, Any]] = []
        accepted_normalized: List[np.ndarray] = []
        for index, (norm_row, physical_row) in enumerate(zip(normalized, physical)):
            failure_code = OK
            if not np.isfinite(norm_row).all() or not np.isfinite(physical_row).all():
                failure_code = NON_FINITE
            elif (
                np.any(norm_row < -1.0 - norm_tolerance)
                or np.any(norm_row > 1.0 + norm_tolerance)
                or np.any(physical_row < lower - physical_tolerance)
                or np.any(physical_row > upper + physical_tolerance)
            ):
                failure_code = OUT_OF_BOUNDS
            elif any(
                np.max(np.abs(norm_row - prior)) <= duplicate_tolerance
                for prior in accepted_normalized
            ):
                failure_code = DUPLICATE
            else:
                accepted_normalized.append(norm_row.copy())
            record: Dict[str, Any] = {
                "candidate_id": int(ids[index]),
                "candidate_valid": failure_code == OK,
                "failure_code": failure_code,
            }
            record.update(
                {f"normalized_{name}": norm_row[i] for i, name in enumerate(self.design_names)}
            )
            record.update(
                {name: physical_row[i] for i, name in enumerate(self.design_names)}
            )
            rows.append(record)
        return pd.DataFrame(rows)

    def validate_candidates(
        self,
        normalized_candidates: np.ndarray,
        physical_candidates: Optional[np.ndarray] = None,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """Validate candidate values without invoking the 0D forward solver."""

        normalized = np.asarray(normalized_candidates, dtype=float)
        physical = (
            np.asarray(self.normalizer.inverse_design(normalized), dtype=float)
            if physical_candidates is None
            else np.asarray(physical_candidates, dtype=float)
        )
        return self._validate_candidate_arrays(
            normalized, physical, candidate_ids=candidate_ids
        )

    def _forward_candidates(
        self,
        checked: pd.DataFrame,
        target: Mapping[str, Any],
    ) -> pd.DataFrame:
        target_thrust = float(target[self.condition_names[0]])
        target_tsfc = float(target[self.condition_names[1]])
        has_mass = "total_mass_kg" in self.condition_names
        target_mass = float(target["total_mass_kg"]) if has_mass else math.nan

        def result_value(result: Any, name: str, default: Any) -> Any:
            return result.get(name, default) if isinstance(result, Mapping) else getattr(result, name, default)

        rows: List[Dict[str, Any]] = []
        for candidate in checked.to_dict(orient="records"):
            record: Dict[str, Any] = dict(candidate)
            record.update(
                {
                    "forward_valid": False,
                    "actual_thrust_N": math.nan,
                    "actual_tsfc_kg_N_s": math.nan,
                    "actual_total_mass_kg": math.nan,
                    "thrust_relative_error": math.nan,
                    "tsfc_relative_error": math.nan,
                    "mass_relative_error": math.nan,
                    "constraint_violation": math.nan,
                    "total_error": math.inf,
                    "within_tolerance": False,
                    "solver_warnings": "",
                }
            )
            if not bool(candidate["candidate_valid"]):
                rows.append(record)
                continue
            try:
                result = self.forward_solver(candidate)
            except Exception as exc:  # solver failures must remain candidate-local
                record["failure_code"] = BRAYTON_EXCEPTION
                record["solver_warnings"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
                continue
            warnings = list(result_value(result, "warnings", []))
            record["solver_warnings"] = " | ".join(str(value) for value in warnings)
            thrust = float(result_value(result, "thrust_N", math.nan))
            tsfc = float(result_value(result, "tsfc_kg_N_s", math.nan))
            mass = float(result_value(result, "total_mass_kg", math.nan))
            record["actual_thrust_N"] = thrust
            record["actual_tsfc_kg_N_s"] = tsfc
            record["actual_total_mass_kg"] = mass
            if not bool(result_value(result, "is_valid", False)):
                record["failure_code"] = INVALID_CYCLE
            elif not math.isfinite(thrust) or thrust <= 0.0:
                record["failure_code"] = NEGATIVE_OR_ZERO_THRUST
            elif not math.isfinite(tsfc) or tsfc <= 0.0:
                record["failure_code"] = NON_FINITE_TSFC
            elif has_mass and (not math.isfinite(mass) or mass <= 0.0):
                record["failure_code"] = NON_FINITE_MASS
            else:
                error_f = abs(thrust - target_thrust) / max(
                    abs(target_thrust), self.derived.thrust_epsilon
                )
                error_t = abs(tsfc - target_tsfc) / max(
                    abs(target_tsfc), self.derived.tsfc_epsilon
                )
                error_m = (
                    abs(mass - target_mass)
                    / max(abs(target_mass), self.derived.mass_epsilon)
                    if has_mass
                    else 0.0
                )
                violation = 0.0
                total = (
                    self.derived.thrust_weight * error_f
                    + self.derived.tsfc_weight * error_t
                    + self.derived.mass_weight * error_m
                    + self.derived.violation_weight * violation
                )
                within = bool(
                    error_f <= self.derived.thrust_relative_tolerance
                    and error_t <= self.derived.tsfc_relative_tolerance
                    and (
                        not has_mass
                        or error_m <= self.derived.mass_relative_tolerance
                    )
                    and total <= self.derived.total_error_tolerance
                )
                record.update(
                    {
                        "forward_valid": True,
                        "failure_code": OK,
                        "thrust_relative_error": error_f,
                        "tsfc_relative_error": error_t,
                        "mass_relative_error": error_m,
                        "constraint_violation": violation,
                        "total_error": total,
                        "within_tolerance": within,
                    }
                )
            rows.append(record)
        return pd.DataFrame(rows)

    def _diversity_selection(self, forward: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        valid = forward.loc[forward["forward_valid"]].copy()
        selected_count = int(self.config["selection"]["selected_count"])
        if valid.empty:
            return pd.DataFrame(columns=list(forward.columns) + ["selected_rank", "selection_reason"]), {
                "valid_candidate_count": 0,
                "selected_count": 0,
                "relaxation_count": 0,
                "final_pool_error_threshold": math.nan,
                "mean_nearest_neighbour_distance": math.nan,
                "minimum_pairwise_distance": math.nan,
                "coverage_fraction_by_dimension": {
                    name: 0.0 for name in self.design_names
                },
            }
        valid = valid.sort_values(["total_error", "candidate_id"], kind="stable")
        selection = self.config["selection"]
        strategy = str(selection.get("strategy", "physics_error_diverse"))
        if strategy == "physics_error_topk":
            selected = valid.head(selected_count).copy()
            selected["selected_rank"] = np.arange(1, len(selected) + 1)
            selected["selection_reason"] = "target_error_topk"
            return selected, {
                "strategy": strategy,
                "valid_candidate_count": len(valid),
                "selected_count": len(selected),
                "relaxation_count": 0,
                "final_pool_error_threshold": (
                    float(selected["total_error"].max()) if len(selected) else math.nan
                ),
                "mean_nearest_neighbour_distance": math.nan,
                "minimum_pairwise_distance": math.nan,
                "coverage_fraction_by_dimension": {},
            }
        if strategy != "physics_error_diverse":
            raise ValueError(f"Unsupported selection strategy: {strategy}")
        threshold = (
            self.derived.total_error_tolerance
            * float(selection["loose_error_multiplier"])
        )
        relaxation_count = 0
        pool = valid.loc[valid["total_error"] <= threshold].copy()
        desired = min(selected_count, len(valid))
        while (
            len(pool) < desired
            and relaxation_count < int(selection["maximum_relaxations"])
        ):
            threshold *= float(selection["relaxation_factor"])
            relaxation_count += 1
            pool = valid.loc[valid["total_error"] <= threshold].copy()

        if pool.empty:
            pool = valid.iloc[[0]].copy()
        desired = min(desired, len(pool))
        normalized_columns = [f"normalized_{name}" for name in self.design_names]
        values = pool[normalized_columns].to_numpy(dtype=float)
        chosen = [0]
        reasons = ["best_error"]
        while len(chosen) < desired:
            remaining = [index for index in range(len(pool)) if index not in chosen]
            distances = np.asarray(
                [
                    min(np.linalg.norm(values[index] - values[prior]) for prior in chosen)
                    for index in remaining
                ]
            )
            best_distance = float(np.max(distances))
            tied = [
                index
                for index, distance in zip(remaining, distances)
                if math.isclose(float(distance), best_distance, rel_tol=0.0, abs_tol=1e-15)
            ]
            chosen.append(tied[0])
            reasons.append("diversity_added")
        selected = pool.iloc[chosen].copy()
        selected["selected_rank"] = np.arange(1, len(selected) + 1)
        selected["selection_reason"] = reasons

        valid_values = valid[normalized_columns].to_numpy(dtype=float)
        if len(valid_values) >= 2:
            tree = cKDTree(valid_values)
            nearest = tree.query(valid_values, k=2, workers=1)[0][:, 1]
            mean_nearest = float(np.mean(nearest))
            pairwise = np.linalg.norm(
                valid_values[:, None, :] - valid_values[None, :, :], axis=2
            )
            pairwise[pairwise == 0.0] = np.inf
            minimum_pairwise = float(np.min(pairwise))
        else:
            mean_nearest = math.nan
            minimum_pairwise = math.nan
        bins = int(self.config["reporting"]["coverage_bins_per_dimension"])
        coverage = {}
        for index, name in enumerate(self.design_names):
            clipped = np.clip(valid_values[:, index], -1.0, 1.0)
            occupied = np.unique(
                np.minimum(((clipped + 1.0) * 0.5 * bins).astype(int), bins - 1)
            )
            coverage[name] = float(len(occupied) / bins)
        metrics = {
            "strategy": strategy,
            "valid_candidate_count": len(valid),
            "selected_count": len(selected),
            "relaxation_count": relaxation_count,
            "final_pool_error_threshold": threshold,
            "mean_nearest_neighbour_distance": mean_nearest,
            "minimum_pairwise_distance": minimum_pairwise,
            "coverage_fraction_by_dimension": coverage,
        }
        return selected, metrics

    def evaluate(
        self,
        target: Mapping[str, Any],
        normalized_candidates: np.ndarray,
        physical_candidates: Optional[np.ndarray] = None,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> RoundTripResult:
        normalized = np.asarray(normalized_candidates, dtype=float)
        if physical_candidates is None:
            physical = np.asarray(self.normalizer.inverse_design(normalized), dtype=float)
        else:
            physical = np.asarray(physical_candidates, dtype=float)
        checked = self._validate_candidate_arrays(
            normalized, physical, candidate_ids=candidate_ids
        )
        forward = self._forward_candidates(checked, target)

        ranked = forward.copy()
        ranked["_valid_order"] = (~ranked["forward_valid"].astype(bool)).astype(int)
        ranked = ranked.sort_values(
            ["_valid_order", "total_error", "candidate_id"], kind="stable"
        ).drop(columns=["_valid_order"])
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        selected, diversity = self._diversity_selection(forward)
        ranked["selected_rank"] = pd.Series(dtype="float64")
        ranked["selection_reason"] = ""
        for row in selected.itertuples(index=False):
            mask = ranked["candidate_id"] == int(row.candidate_id)
            ranked.loc[mask, "selected_rank"] = int(row.selected_rank)
            ranked.loc[mask, "selection_reason"] = str(row.selection_reason)

        valid_ranked = ranked.loc[ranked["forward_valid"]]
        top_k_metrics: Dict[str, Any] = {}
        for requested in [int(value) for value in self.config["reporting"]["top_k"]]:
            effective = min(requested, len(valid_ranked))
            head = valid_ranked.head(effective)
            top_k_metrics[str(requested)] = {
                "requested_k": requested,
                "effective_k": effective,
                "success": bool(effective > 0 and head["within_tolerance"].any()),
                "best_total_error": (
                    float(head["total_error"].min()) if effective > 0 else math.nan
                ),
            }

        is_in_domain = self.target_in_training_domain(target)
        expected_status = str(target.get("expected_status", "ordinary"))
        target_type = str(target.get("target_type", "unknown"))
        any_success = bool(valid_ranked["within_tolerance"].any())
        is_ood_target = target_type.lower() in {
            "ood",
            "ood-feasible-unknown",
        }
        is_rejection_target = (
            expected_status.lower() in {"reject", "rejection"}
            or target_type.lower() == "infeasible"
        )
        abstention = {
            "is_training_domain_target": is_in_domain,
            "is_ood_target": is_ood_target,
            "is_rejection_target": is_rejection_target,
            "accepted": any_success,
            "rejected": not any_success,
            "false_accept": bool(is_rejection_target and any_success),
            "low_confidence": bool(not any_success),
        }
        failure_rows = forward.loc[forward["failure_code"] != OK, [
            "candidate_id", "failure_code", "solver_warnings"
        ]].copy()
        if not any_success:
            failure_rows = pd.concat(
                [
                    failure_rows,
                    pd.DataFrame(
                        [
                            {
                                "candidate_id": -1,
                                "failure_code": NO_CANDIDATE_WITHIN_TOLERANCE,
                                "solver_warnings": "target-level outcome",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        counts = forward["failure_code"].value_counts().to_dict()
        metrics = {
            "metrics_version": self.derived.metrics_version,
            "candidate_count": len(forward),
            "candidate_valid_count": int(checked["candidate_valid"].sum()),
            "forward_valid_count": int(forward["forward_valid"].sum()),
            "duplicate_count": int((checked["failure_code"] == DUPLICATE).sum()),
            "duplicate_rate": float((checked["failure_code"] == DUPLICATE).mean()) if len(checked) else 0.0,
            "failure_counts": {str(key): int(value) for key, value in counts.items()},
            "top_k": top_k_metrics,
            "diversity": diversity,
            "abstention": abstention,
            "best": {
                "candidate_id": int(valid_ranked.iloc[0]["candidate_id"]) if len(valid_ranked) else None,
                "thrust_relative_error": float(valid_ranked.iloc[0]["thrust_relative_error"]) if len(valid_ranked) else math.nan,
                "tsfc_relative_error": float(valid_ranked.iloc[0]["tsfc_relative_error"]) if len(valid_ranked) else math.nan,
                "mass_relative_error": float(valid_ranked.iloc[0]["mass_relative_error"]) if len(valid_ranked) else math.nan,
                "total_error": float(valid_ranked.iloc[0]["total_error"]) if len(valid_ranked) else math.nan,
                "within_tolerance": bool(valid_ranked.iloc[0]["within_tolerance"]) if len(valid_ranked) else False,
            },
        }
        return RoundTripResult(
            checked_candidates=checked,
            forward_results=forward,
            ranked_candidates=ranked,
            failures=failure_rows,
            metrics=metrics,
        )


def target_out_of_domain_result(
    target: Mapping[str, Any],
    model: str,
) -> RoundTripResult:
    """Create a traceable target-level rejection when sampling is impossible."""

    target_type = str(target.get("target_type", "")).lower()
    expected_status = str(target.get("expected_status", "")).lower()
    is_ood_target = target_type in {"ood", "ood-feasible-unknown"}
    is_rejection_target = (
        expected_status in {"reject", "rejection"} or target_type == "infeasible"
    )
    empty = pd.DataFrame()
    failure = pd.DataFrame(
        [
            {
                "candidate_id": -1,
                "failure_code": TARGET_OUT_OF_DOMAIN,
                "solver_warnings": "target cannot be normalized for generator sampling",
            }
        ]
    )
    metrics = {
        "candidate_count": 0,
        "candidate_valid_count": 0,
        "forward_valid_count": 0,
        "duplicate_count": 0,
        "duplicate_rate": 0.0,
        "failure_counts": {TARGET_OUT_OF_DOMAIN: 1},
        "top_k": {},
        "diversity": {"valid_candidate_count": 0, "selected_count": 0},
        "abstention": {
            "is_training_domain_target": False,
            "is_ood_target": is_ood_target,
            "is_rejection_target": is_rejection_target,
            "accepted": False,
            "rejected": True,
            "false_accept": False,
            "low_confidence": True,
        },
        "best": {
            "candidate_id": None,
            "thrust_relative_error": math.nan,
            "tsfc_relative_error": math.nan,
            "mass_relative_error": math.nan,
            "total_error": math.nan,
            "within_tolerance": False,
        },
        "model": model,
        "target_id": target.get("target_id"),
    }
    return RoundTripResult(empty, empty, empty, failure, metrics)


def sampling_budget_exhausted_result(
    target: Mapping[str, Any],
    model: str,
    *,
    is_training_domain: bool,
) -> RoundTripResult:
    """Record a completed rejection-sampling attempt with no accepted design."""

    target_type = str(target.get("target_type", "")).lower()
    expected_status = str(target.get("expected_status", "")).lower()
    is_ood_target = target_type in {"ood", "ood-feasible-unknown"}
    is_rejection_target = (
        expected_status in {"reject", "rejection"} or target_type == "infeasible"
    )
    empty = pd.DataFrame()
    failure = pd.DataFrame(
        [
            {
                "candidate_id": -1,
                "failure_code": SAMPLING_BUDGET_EXHAUSTED,
                "solver_warnings": (
                    "bounded rejection sampling reached its maximum network-draw "
                    "budget without an in-bound candidate"
                ),
            }
        ]
    )
    metrics = {
        "candidate_count": 0,
        "candidate_valid_count": 0,
        "forward_valid_count": 0,
        "duplicate_count": 0,
        "duplicate_rate": 0.0,
        "failure_counts": {SAMPLING_BUDGET_EXHAUSTED: 1},
        "top_k": {},
        "diversity": {"valid_candidate_count": 0, "selected_count": 0},
        "abstention": {
            "is_training_domain_target": bool(is_training_domain),
            "is_ood_target": is_ood_target,
            "is_rejection_target": is_rejection_target,
            "accepted": False,
            "rejected": True,
            "false_accept": False,
            "low_confidence": True,
        },
        "best": {
            "candidate_id": None,
            "thrust_relative_error": math.nan,
            "tsfc_relative_error": math.nan,
            "mass_relative_error": math.nan,
            "total_error": math.nan,
            "within_tolerance": False,
        },
        "model": model,
        "target_id": target.get("target_id"),
    }
    return RoundTripResult(empty, empty, empty, failure, metrics)
