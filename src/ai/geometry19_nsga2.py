"""True-physics warm-start NSGA-II for WP19-23E."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from src.ai.geometry19_inverse_contract import load_target
from src.ai.geometry19_pipeline_context import PipelineContext, resolve_pipeline_context
from src.ai.geometry19_physics import evaluate_geometry19_candidate
from src.ai.geometry19_selection import (
    ConstraintDominanceContract,
    attach_dispositions,
    build_disposition_rows,
    candidate_sort_key,
    constraint_state,
    rank_delivery_candidates,
)
from src.ai.geometry19_surrogate import sha256_file
from src.physics import Environment


class Geometry19NSGA2Error(ValueError):
    """Raised when WP19-23E violates its true-physics optimizer contract."""


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
            stream.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Geometry19NSGA2Error(f"Cannot read ledger: {path}") from exc


def dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right) and np.any(left < right))


def constraint_dominates(
    left: np.ndarray,
    right: np.ndarray,
    left_feasible: bool,
    right_feasible: bool,
    left_violation: float,
    right_violation: float,
) -> bool:
    """Apply the frozen Deb feasibility rules before objective dominance."""

    if left_feasible != right_feasible:
        return left_feasible
    if not left_feasible:
        return left_violation < right_violation
    return dominates(left, right)


def nondominated_fronts(
    objectives: np.ndarray,
    violations: np.ndarray | None = None,
    feasible: np.ndarray | None = None,
) -> list[list[int]]:
    count = len(objectives)
    if violations is None:
        violations = np.zeros(count, dtype=float)
    if feasible is None:
        feasible = np.ones(count, dtype=bool)
    violations = np.asarray(violations, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    if violations.shape != (count,) or feasible.shape != (count,):
        raise Geometry19NSGA2Error("Constraint dominance vectors have invalid shapes")
    domination_count = np.zeros(count, dtype=int)
    dominated: list[list[int]] = [[] for _ in range(count)]
    fronts: list[list[int]] = [[]]
    for left in range(count):
        for right in range(left + 1, count):
            if constraint_dominates(
                objectives[left],
                objectives[right],
                bool(feasible[left]),
                bool(feasible[right]),
                float(violations[left]),
                float(violations[right]),
            ):
                dominated[left].append(right)
                domination_count[right] += 1
            elif constraint_dominates(
                objectives[right],
                objectives[left],
                bool(feasible[right]),
                bool(feasible[left]),
                float(violations[right]),
                float(violations[left]),
            ):
                dominated[right].append(left)
                domination_count[left] += 1
        if domination_count[left] == 0:
            fronts[0].append(left)
    position = 0
    while position < len(fronts) and fronts[position]:
        following = []
        for left in fronts[position]:
            for right in dominated[left]:
                domination_count[right] -= 1
                if domination_count[right] == 0:
                    following.append(right)
        if following:
            fronts.append(following)
        position += 1
    return fronts


def hit_aligned_fronts(
    objectives: np.ndarray,
    violations: np.ndarray,
    feasible: np.ndarray,
) -> list[list[int]]:
    """Rank by failed axes, rho, sum, then engineering violation."""

    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise Geometry19NSGA2Error("Hit-aligned objectives must have shape [n,3]")
    if violations.shape != (len(objectives),) or feasible.shape != (len(objectives),):
        raise Geometry19NSGA2Error("Hit-aligned constraint arrays have invalid shape")

    def key(index: int) -> tuple[Any, ...]:
        values = objectives[index]
        return (
            int(np.sum(values > 1.0)),
            float(np.max(values)),
            float(np.sum(values)),
            not bool(feasible[index]),
            float(violations[index]),
            int(index),
        )

    return [[index] for index in sorted(range(len(objectives)), key=key)]


def crowding_distance(objectives: np.ndarray, front: Sequence[int]) -> np.ndarray:
    size = len(front)
    distances = np.zeros(size, dtype=float)
    if size <= 2:
        distances[:] = math.inf
        return distances
    indices = np.asarray(front, dtype=int)
    for objective in range(objectives.shape[1]):
        order = np.argsort(objectives[indices, objective], kind="stable")
        distances[order[0]] = math.inf
        distances[order[-1]] = math.inf
        span = objectives[indices[order[-1]], objective] - objectives[indices[order[0]], objective]
        if span <= 1.0e-15:
            continue
        for position in range(1, size - 1):
            distances[order[position]] += (
                objectives[indices[order[position + 1]], objective]
                - objectives[indices[order[position - 1]], objective]
            ) / span
    return distances


def _objectives(evaluation: Mapping[str, Any]) -> np.ndarray:
    hit_alignment = evaluation.get("hit_alignment")
    if evaluation.get("physics_acceptable") and isinstance(hit_alignment, Mapping):
        ratios = hit_alignment.get("axis_ratios")
        if isinstance(ratios, Mapping):
            return np.asarray(
                [ratios["thrust_N"], ratios["tsfc_kg_N_s"], ratios["mass_kg"]],
                dtype=float,
            )
    errors = evaluation.get("relative_errors")
    if not evaluation.get("physics_acceptable") or not isinstance(errors, Mapping):
        return np.full(3, 1.0e6, dtype=float)
    return np.asarray(
        [errors["thrust_N"], errors["tsfc_kg_N_s"], errors["mass_kg"]], dtype=float
    )


def _constraint_arrays(
    evaluations: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    states = [constraint_state(row) for row in evaluations]
    return (
        np.asarray([state[1] for state in states], dtype=float),
        np.asarray([state[0] for state in states], dtype=bool),
    )


def _front_crowding(
    objectives: np.ndarray,
    violations: np.ndarray,
    feasible: np.ndarray,
    front: Sequence[int],
) -> np.ndarray:
    values = objectives if all(bool(feasible[index]) for index in front) else violations[:, None]
    return crowding_distance(values, front)


def _rank_and_crowding(population: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    evaluations = [item["evaluation"] for item in population]
    objectives = np.asarray([_objectives(row) for row in evaluations])
    violations, feasible = _constraint_arrays(evaluations)
    ranks = np.empty(len(population), dtype=int)
    crowding = np.zeros(len(population), dtype=float)
    for rank, front in enumerate(hit_aligned_fronts(objectives, violations, feasible)):
        ranks[front] = rank
        values = _front_crowding(objectives, violations, feasible, front)
        for position, index in enumerate(front):
            crowding[index] = values[position]
    return ranks, crowding


def _environmental_selection(
    population: Sequence[Mapping[str, Any]], size: int
) -> list[dict[str, Any]]:
    evaluations = [item["evaluation"] for item in population]
    objectives = np.asarray([_objectives(row) for row in evaluations])
    violations, feasible = _constraint_arrays(evaluations)
    selected: list[int] = []
    for front in nondominated_fronts(objectives, violations, feasible):
        if len(selected) + len(front) <= size:
            selected.extend(front)
            continue
        distances = _front_crowding(objectives, violations, feasible, front)
        order = sorted(
            range(len(front)),
            key=lambda position: (-distances[position], front[position]),
        )
        selected.extend(front[position] for position in order[: size - len(selected)])
        break
    return [dict(population[index]) for index in selected]


def _lhs_fill(
    count: int,
    lower: np.ndarray,
    upper: np.ndarray,
    integer_indices: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, len(lower)), dtype=float)
    unit = np.empty((count, len(lower)), dtype=float)
    for dimension in range(len(lower)):
        unit[:, dimension] = (rng.permutation(count) + rng.random(count)) / count
    values = lower + unit * (upper - lower)
    values[:, integer_indices] = np.rint(values[:, integer_indices])
    return np.clip(values, lower, upper)


def power_of_two_population_size(
    warm_start_count: int,
    minimum: int,
    maximum: int,
) -> int:
    """Choose the smallest allowed power of two that can contain warm starts."""

    if warm_start_count <= 0:
        raise Geometry19NSGA2Error("Warm-start population requires at least one design")
    if minimum <= 0 or maximum < minimum:
        raise Geometry19NSGA2Error("Invalid NSGA-II population bounds")
    if minimum & (minimum - 1) or maximum & (maximum - 1):
        raise Geometry19NSGA2Error("NSGA-II population bounds must be powers of two")
    desired = 1 << (int(warm_start_count) - 1).bit_length()
    return min(max(desired, minimum), maximum)


def build_initial_population(
    warm_rows: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    population_size: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    names = [item["name"] for item in fields]
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=float)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=float)
    integer_indices = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    ordered = sorted(
        (row for row in warm_rows if row.get("physics_acceptable")),
        key=candidate_sort_key,
    )
    genes = []
    provenance = []
    seen = set()
    for row in ordered:
        design_hash = str(row["design_hash"])
        if design_hash in seen:
            continue
        vector = np.asarray([row["geometry19"][name] for name in names], dtype=float)
        if np.any(vector < lower) or np.any(vector > upper):
            continue
        vector[integer_indices] = np.rint(vector[integer_indices])
        genes.append(vector)
        provenance.append(
            {
                "source": "true_physics_warm_start",
                "source_evaluation_id": row["evaluation_id"],
                "source_total_true_error": row["total_true_error"],
            }
        )
        seen.add(design_hash)
        if len(genes) == population_size:
            break
    warm_count = len(genes)
    rng = np.random.default_rng(seed)
    fill = _lhs_fill(population_size - warm_count, lower, upper, integer_indices, rng)
    for index, vector in enumerate(fill):
        genes.append(vector)
        provenance.append({"source": "lhs_fill", "source_row": index, "seed": seed})
    return np.asarray(genes, dtype=float), provenance


def _tournament(
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
    size: int = 3,
) -> int:
    choices = rng.choice(len(ranks), size=min(size, len(ranks)), replace=False)
    return int(
        min(
            choices,
            key=lambda index: (ranks[index], -crowding[index], int(index)),
        )
    )


def _sbx(
    left: np.ndarray,
    right: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    rate: float,
    eta: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    first = left.copy()
    second = right.copy()
    if rng.random() >= rate:
        return first, second
    for index in range(len(left)):
        if rng.random() >= 0.5 or abs(left[index] - right[index]) <= 1.0e-12:
            continue
        value = rng.random()
        beta = (2.0 * value) ** (1.0 / (eta + 1.0)) if value <= 0.5 else (
            1.0 / (2.0 * (1.0 - value))
        ) ** (1.0 / (eta + 1.0))
        first[index] = 0.5 * ((1.0 + beta) * left[index] + (1.0 - beta) * right[index])
        second[index] = 0.5 * ((1.0 - beta) * left[index] + (1.0 + beta) * right[index])
    return np.clip(first, lower, upper), np.clip(second, lower, upper)


def _mutate(
    genes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    integer_indices: Sequence[int],
    rng: np.random.Generator,
    rate: float,
    scale: float,
) -> np.ndarray:
    result = genes.copy()
    mask = rng.random(len(result)) < rate
    result[mask] += rng.normal(0.0, scale, int(mask.sum())) * (upper - lower)[mask]
    result = np.clip(result, lower, upper)
    result[list(integer_indices)] = np.rint(result[list(integer_indices)])
    return result


def _design_mapping(genes: np.ndarray, fields: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        item["name"]: int(round(float(genes[index])))
        if item["dtype"] == "int64"
        else float(genes[index])
        for index, item in enumerate(fields)
    }


def _design_hash(mapping: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _reconstruct_warm_rows(
    physics_root: Path, refinement_root: Path
) -> list[dict[str, Any]]:
    physics_rows = _read_jsonl(physics_root / "physics_forward_ledger.jsonl")
    by_id = {row["evaluation_id"]: row for row in physics_rows}
    pool = [row for row in physics_rows if row["variant"] in {"direct", "dflow"}]
    ledger = _read_jsonl(refinement_root / "refinement_ledger.jsonl")
    current = {}
    for row in ledger:
        seed = int(row["seed_index"])
        if seed not in current:
            current[seed] = by_id.get(row["seed_evaluation_id"])
        if row["accepted"]:
            current[seed] = row["proposed"]
    pool.extend(row for row in current.values() if row is not None)
    return pool


def run_nsga2_stage(
    repository_root: str | Path,
    target_path: str | Path,
    *,
    physics_root: str | Path | None = None,
    refinement_root: str | Path | None = None,
    candidate_root: str | Path | None = None,
    sampling_mode: str | None = None,
    cold_start: bool = False,
    cold_start_comparison: bool = False,
    output_root: str | Path | None = None,
    pipeline_context: PipelineContext | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target_source = Path(target_path).resolve()
    target = load_target(target_source)
    context = resolve_pipeline_context(root, pipeline_context)
    contract_path = context.contract_path
    contract = context.contract
    protocol = context.protocol
    physics_runtime = context.physics_runtime
    base = root / protocol["artifact_protocol"]["runtime_root"] / target.target_id
    physics_source = Path(physics_root).resolve() if physics_root else base / "wp19_23c"
    refinement_source = (
        Path(refinement_root).resolve() if refinement_root else base / "wp19_23d"
    )
    candidate_source = Path(candidate_root).resolve() if candidate_root else None
    runtime = Path(output_root).resolve() if output_root else base / "wp19_23e"
    if runtime.exists():
        raise Geometry19NSGA2Error(f"Refusing to overwrite NSGA-II run: {runtime}")
    runtime.mkdir(parents=True)
    try:
        optimizer = protocol["optimizer"]
        minimum_population = int(optimizer["minimum_population_size"])
        maximum_population = int(optimizer["maximum_population_size"])
        maximum_generations = int(optimizer["maximum_generations"])
        maximum = int(optimizer["maximum_physics_evaluations"])
        if maximum != 640:
            raise Geometry19NSGA2Error("WP19-23E requires a 640-evaluation budget")
        schema_path = physics_runtime.source_paths["design_schema"]
        schema = physics_runtime.design_schema
        fields = list(schema["design_variables"])
        lower = np.asarray([item["bounds"][0] for item in fields], dtype=float)
        upper = np.asarray([item["bounds"][1] for item in fields], dtype=float)
        integer_indices = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
        source_summary_path: Path
        source_calibration_hash: str
        if cold_start and cold_start_comparison:
            raise Geometry19NSGA2Error(
                "cold_start and cold_start_comparison are mutually exclusive"
            )
        uses_cold_start = cold_start or cold_start_comparison
        if cold_start_comparison:
            if candidate_source is not None or sampling_mode is not None:
                raise Geometry19NSGA2Error(
                    "Cold-start comparison must not use CFM candidate artifacts"
                )
            source_summary_path = runtime / "cold_start_comparison_contract.json"
            _write_json(
                source_summary_path,
                {
                    "contract_version": "wp19_23_cold_start_nsga2_comparison_v1",
                    "target_id": target.target_id,
                    "initialization": "latin_hypercube_geometry19",
                    "optimizer_seed": int(optimizer["seed"]),
                    "maximum_physics_evaluations": maximum,
                    "formal_use_allowed": False,
                },
            )
            population_size = int(optimizer["cold_start_population_size"])
            if population_size != power_of_two_population_size(
                population_size, minimum_population, maximum_population
            ):
                raise Geometry19NSGA2Error(
                    "Cold-start comparison population must be an allowed power of two"
                )
            available_warm_count = 0
            warm_rows = []
            source_calibration_hash = physics_runtime.model_config.calibration_pack_hash
            initialization_mode = "cold_start_lhs_comparator"
        elif cold_start:
            if candidate_source is None or sampling_mode not in {"direct", "dflow"}:
                raise Geometry19NSGA2Error(
                    "Cold-start requires candidate_root and direct/dflow sampling_mode"
                )
            source_summary_path = candidate_source / "summary.json"
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            if (
                source_summary["target_id"] != target.target_id
                or source_summary["sampling_mode"] != sampling_mode
            ):
                raise Geometry19NSGA2Error("Cold-start candidate summary mismatch")
            if int(source_summary["candidate_protocol"]["n_valid"]) != 0:
                raise Geometry19NSGA2Error("Cold-start is only allowed when N_valid=0")
            population_size = int(optimizer["cold_start_population_size"])
            if population_size != power_of_two_population_size(
                population_size, minimum_population, maximum_population
            ):
                raise Geometry19NSGA2Error("Cold-start population must be an allowed power of two")
            available_warm_count = 0
            warm_rows: list[dict[str, Any]] = []
            source_calibration_hash = str(source_summary["hashes"]["calibration_hash"])
            initialization_mode = "cold_start_lhs"
        else:
            refinement_summary_path = refinement_source / "summary.json"
            refinement_summary = json.loads(
                refinement_summary_path.read_text(encoding="utf-8")
            )
            if refinement_summary["target_id"] != target.target_id:
                raise Geometry19NSGA2Error("WP19-23D target mismatch")
            if refinement_summary["downstream_stop"]:
                raise Geometry19NSGA2Error("WP19-23D already hit; NSGA-II is forbidden")
            source_summary_path = refinement_summary_path
            source_calibration_hash = str(refinement_summary["hashes"]["calibration_hash"])
            warm_rows = _reconstruct_warm_rows(physics_source, refinement_source)
            _, maximum_provenance = build_initial_population(
                warm_rows, fields, maximum_population, int(optimizer["seed"])
            )
            available_warm_count = sum(
                row["source"] == "true_physics_warm_start"
                for row in maximum_provenance
            )
            population_size = power_of_two_population_size(
                available_warm_count, minimum_population, maximum_population
            )
            initialization_mode = "warm_start"
        generations = min(
            maximum_generations,
            maximum // population_size - 1,
        )
        if generations < 1:
            raise Geometry19NSGA2Error("Physics budget cannot fund one NSGA-II generation")
        genes, provenance = build_initial_population(
            warm_rows, fields, population_size, int(optimizer["seed"])
        )
        if uses_cold_start:
            provenance = [
                {**row, "source": "cold_start_lhs", "cold_start": True}
                for row in provenance
            ]
        warm_count = sum(row["source"] == "true_physics_warm_start" for row in provenance)
        if not uses_cold_start and warm_count == 0:
            raise Geometry19NSGA2Error("No valid warm-start design exists")
        model_config = physics_runtime.model_config
        constraint_profile = physics_runtime.constraint_profile
        dominance_contract = context.constraint_dominance
        if model_config.calibration_pack_hash != source_calibration_hash:
            raise Geometry19NSGA2Error("NSGA-II calibration hash mismatch")
        source_contract = json.loads(source_summary_path.read_text(encoding="utf-8"))
        source_physics_bundle_hash = (source_contract.get("hashes") or {}).get(
            "physics_bundle_hash"
        )
        if (
            source_physics_bundle_hash is not None
            and source_physics_bundle_hash != physics_runtime.physics_bundle_hash
        ):
            raise Geometry19NSGA2Error("NSGA-II physics bundle mismatch")
        rng = np.random.default_rng(int(optimizer["seed"]))
        cache: dict[str, dict[str, Any]] = {}
        ledger: list[dict[str, Any]] = []
        logical_evaluations = 0
        unique_calls = 0

        def evaluate_population(
            matrix: np.ndarray, generation: int, sources: Sequence[Mapping[str, Any]]
        ) -> list[dict[str, Any]]:
            nonlocal logical_evaluations, unique_calls
            population = []
            for index, vector in enumerate(matrix):
                mapping = _design_mapping(vector, fields)
                design_hash = _design_hash(mapping)
                cache_key = f"{physics_runtime.physics_bundle_hash}:{design_hash}"
                cache_hit = cache_key in cache
                if not cache_hit:
                    cache[cache_key] = evaluate_geometry19_candidate(
                        logical_evaluations,
                        f"nsga2_generation_{generation}_individual_{index}",
                        mapping,
                        design_hash,
                        target,
                        contract.diagnostic_acceptance,
                        model_config,
                        constraint_profile,
                        Environment(),
                        physics_runtime=physics_runtime,
                        constraint_dominance_contract=dominance_contract,
                    )
                    unique_calls += 1
                evaluation = dict(cache[cache_key])
                evaluation["candidate_id"] = logical_evaluations
                evaluation["evaluation_id"] = (
                    f"nsga2_generation_{generation}_individual_{index}:{logical_evaluations}"
                )
                evaluation["variant"] = "nsga2"
                record = {
                    "evaluation_index": logical_evaluations,
                    "generation": generation,
                    "population_index": index,
                    "cache_hit": cache_hit,
                    "provenance": dict(sources[index]),
                    "genes": mapping,
                    "evaluation": evaluation,
                }
                ledger.append(record)
                population.append(
                    {
                        "genes": np.asarray(vector, dtype=float),
                        "evaluation": evaluation,
                        "provenance": dict(sources[index]),
                    }
                )
                logical_evaluations += 1
            return population

        population = evaluate_population(genes, 0, provenance)
        stopped_generation = 0 if any(
            item["evaluation"]["development_diagnostic_hit"] for item in population
        ) else None
        for generation in range(1, generations + 1):
            if stopped_generation is not None:
                break
            ranks, crowding = _rank_and_crowding(population)
            children = []
            child_provenance = []
            while len(children) < population_size:
                left_index = _tournament(ranks, crowding, rng)
                right_index = _tournament(ranks, crowding, rng)
                first, second = _sbx(
                    population[left_index]["genes"],
                    population[right_index]["genes"],
                    lower,
                    upper,
                    rng,
                    rate=0.9,
                )
                for child in (first, second):
                    child = _mutate(
                        child,
                        lower,
                        upper,
                        integer_indices,
                        rng,
                        rate=0.1,
                        scale=0.1,
                    )
                    children.append(child)
                    child_provenance.append(
                        {
                            "source": "nsga2_offspring",
                            "parent_population_indices": [left_index, right_index],
                        }
                    )
                    if len(children) == population_size:
                        break
            offspring = evaluate_population(
                np.asarray(children), generation, child_provenance
            )
            population = _environmental_selection(population + offspring, population_size)
            if any(
                item["evaluation"]["development_diagnostic_hit"] for item in offspring
            ):
                stopped_generation = generation
        if logical_evaluations > maximum:
            raise Geometry19NSGA2Error("NSGA-II exceeded the 640-evaluation budget")
        all_evaluations = [row["evaluation"] for row in ledger]
        hits = [row for row in all_evaluations if row["development_diagnostic_hit"]]
        valid = [row for row in all_evaluations if row["physics_acceptable"]]
        valid.sort(key=candidate_sort_key)
        valid_objectives = np.asarray([_objectives(row) for row in valid])
        valid_violations, valid_feasible = _constraint_arrays(valid)
        pareto = (
            [
                valid[index]
                for index in hit_aligned_fronts(
                    valid_objectives, valid_violations, valid_feasible
                )[0]
            ]
            if valid
            else []
        )
        pareto.sort(key=candidate_sort_key)
        delivery_pool = hits if hits else pareto
        delivered = rank_delivery_candidates(
            delivery_pool, int(protocol["delivery"]["maximum_candidates"]), fields
        )
        dispositions = build_disposition_rows(
            all_evaluations,
            delivered,
            stage="WP19-23E",
            next_stage=None,
            contract=dominance_contract,
        )
        attach_dispositions(all_evaluations, dispositions)
        attach_dispositions(delivered, dispositions)
        summary = {
            "work_package": "WP19-23E",
            "status": "NSGA2_DIAGNOSTIC_HIT" if hits else "BEST_APPROXIMATION_NOT_HIT",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target.target_id,
            "initialization_mode": initialization_mode,
            "comparison_only": bool(cold_start_comparison),
            "formal_use_allowed": False,
            "formal_hit": None,
            "downstream_stop": True,
            "next_stage": "DELIVERY_AND_VISUALIZATION",
            "stopped_after_generation": stopped_generation,
            "counts": {
                "population_size": population_size,
                "available_warm_start_count": available_warm_count,
                "maximum_configured_generations": maximum_generations,
                "budget_limited_generations": generations,
                "configured_generations": generations,
                "planned_maximum_physics_evaluations": population_size
                * (generations + 1),
                "completed_generations": (
                    stopped_generation if stopped_generation is not None else generations
                ),
                "warm_start_count": warm_count,
                "lhs_fill_count": population_size - warm_count,
                "logical_physics_evaluations": logical_evaluations,
                "unique_v2_1_evaluations": unique_calls,
                "unique_physics_evaluations": unique_calls,
                "diagnostic_hits": len(hits),
                "pareto_size": len(pareto),
                "delivered_count": len(delivered),
                "constraint_feasible": sum(
                    bool((row.get("constraint_dominance") or {}).get("constraint_feasible"))
                    for row in all_evaluations
                ),
            },
            "best_approximation": valid[0] if valid else None,
            "delivered_evaluation_ids": [row["evaluation_id"] for row in delivered],
            "disposition_counts": {
                reason: sum(row["disposition_reason"] == reason for row in dispositions)
                for reason in sorted({row["disposition_reason"] for row in dispositions})
            },
            "hashes": {
                "protocol_hash": sha256_file(contract_path),
                "target_hash": sha256_file(target_source),
                "schema_hash": sha256_file(schema_path),
                "calibration_hash": model_config.calibration_pack_hash,
                "physics_backend_id": physics_runtime.backend_id,
                "physics_bundle_hash": physics_runtime.physics_bundle_hash,
                "constraint_profile_hash": constraint_profile.source_hash,
                "constraint_dominance_hash": dominance_contract.source_hash,
                "initialization_source_summary_hash": sha256_file(source_summary_path),
                "code_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            },
        }
        _write_json(runtime / "target.json", target.as_mapping())
        _write_jsonl(runtime / "nsga2_ledger.jsonl", ledger)
        _write_jsonl(runtime / "candidate_disposition_ledger.jsonl", dispositions)
        _write_json(runtime / "pareto_candidates.json", pareto)
        _write_json(runtime / "delivered_candidates.json", delivered)
        _write_json(runtime / "summary.json", summary)
        _write_json(
            runtime / "manifest.json",
            {
                "protocol_version": protocol["protocol_version"],
                "target_id": target.target_id,
                "stage": "WP19-23E",
                "required_files": [
                    "manifest.json",
                    "target.json",
                    "nsga2_ledger.jsonl",
                    "candidate_disposition_ledger.jsonl",
                    "pareto_candidates.json",
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
