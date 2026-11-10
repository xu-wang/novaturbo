"""Named 5D-to-19D population adaptation for WP5 Warm-Start."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .dataset import DesignSpaceBounds
from .optimizer import InitialPopulation, optimizer_parameter_space


WARM_START_MODES = ("random", "generator", "hybrid")
BACKGROUND_METHODS = ("random", "lhs")


@dataclass(frozen=True)
class WarmStartBuildResult:
    population: InitialPopulation
    provenance: pd.DataFrame
    metadata: Dict[str, Any]


def _sample_unit_background(
    population_size: int,
    n_params: int,
    seed: int,
    method: str,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if method == "random":
        return rng.random((population_size, n_params))
    if method != "lhs":
        raise ValueError(f"background_method must be one of {BACKGROUND_METHODS}")
    unit = np.zeros((population_size, n_params), dtype=float)
    intervals = np.linspace(0.0, 1.0, population_size + 1)
    for dimension in range(n_params):
        points = (
            intervals[:-1]
            + rng.uniform(0.0, 1.0, population_size) / population_size
        )
        rng.shuffle(points)
        unit[:, dimension] = points
    return unit


def build_background_population(
    bounds: Optional[DesignSpaceBounds] = None,
    population_size: int = 100,
    seed: int = 42,
    method: str = "random",
) -> Tuple[np.ndarray, List[str]]:
    """Build the complete 19D background shared by Random and Hybrid."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    bounds = bounds or DesignSpaceBounds()
    param_names, lower, upper = optimizer_parameter_space(bounds)
    unit = _sample_unit_background(population_size, len(param_names), seed, method)
    return lower + unit * (upper - lower), param_names


def _candidate_matrix(
    generated_candidates: Optional[Any],
    generated_field_names: Sequence[str],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if generated_candidates is None:
        return np.empty((0, len(generated_field_names)), dtype=float), []
    if isinstance(generated_candidates, pd.DataFrame):
        missing = set(generated_field_names).difference(generated_candidates.columns)
        if missing:
            raise ValueError(
                f"generated candidate table is missing fields: {sorted(missing)}"
            )
        matrix = generated_candidates[list(generated_field_names)].to_numpy(dtype=float)
        row_metadata = []
        for source_index, row in generated_candidates.reset_index(drop=True).iterrows():
            metadata: Dict[str, Any] = {"generated_source_row": int(source_index)}
            for key in ("candidate_id", "rank", "selected_rank", "selection_reason"):
                if key in row and not pd.isna(row[key]):
                    value = row[key]
                    metadata[key] = value.item() if isinstance(value, np.generic) else value
            row_metadata.append(metadata)
    else:
        matrix = np.asarray(generated_candidates, dtype=float)
        row_metadata = [
            {"generated_source_row": index} for index in range(len(matrix))
        ] if matrix.ndim >= 1 else []
    if matrix.ndim != 2 or matrix.shape[1] != len(generated_field_names):
        raise ValueError(
            "generated candidates must have shape "
            f"[n,{len(generated_field_names)}], got {matrix.shape}"
        )
    return matrix, row_metadata


def _validate_and_deduplicate_candidates(
    matrix: np.ndarray,
    row_metadata: List[Dict[str, Any]],
    generated_field_names: Sequence[str],
    param_names: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    duplicate_tolerance: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[int]]:
    if duplicate_tolerance < 0.0:
        raise ValueError("duplicate_tolerance must be non-negative")
    if not np.isfinite(matrix).all():
        locations = np.argwhere(~np.isfinite(matrix))[:8].tolist()
        raise ValueError(f"generated candidates contain NaN/inf at indices {locations}")
    indices = [param_names.index(name) for name in generated_field_names]
    candidate_lower = lower[indices]
    candidate_upper = upper[indices]
    out_of_bounds = (matrix < candidate_lower) | (matrix > candidate_upper)
    if out_of_bounds.any():
        locations = np.argwhere(out_of_bounds)[:8].tolist()
        raise ValueError(
            f"generated candidates are outside DesignSpaceBounds at indices {locations}"
        )
    normalized = (matrix - candidate_lower) / (candidate_upper - candidate_lower)
    keep: List[int] = []
    duplicates: List[int] = []
    for index, row in enumerate(normalized):
        if any(
            np.max(np.abs(row - normalized[prior])) <= duplicate_tolerance
            for prior in keep
        ):
            duplicates.append(index)
        else:
            keep.append(index)
    return (
        matrix[keep],
        [row_metadata[index] for index in keep],
        duplicates,
    )


def build_warm_start_population(
    generated_candidates: Optional[Any],
    generated_field_names: Sequence[str],
    *,
    mode: str,
    population_size: int = 100,
    generator_ratio: Optional[float] = None,
    background_seed: int = 42,
    background_method: str = "random",
    bounds: Optional[DesignSpaceBounds] = None,
    duplicate_tolerance: float = 1.0e-9,
    candidate_source: str = "generator",
) -> WarmStartBuildResult:
    """Overlay named generator fields on a frozen complete 19D background.

    Input row order is treated as the upstream ranking order.  If too many
    generated rows are supplied, the prefix is selected.  If too few remain
    after 5D de-duplication, untouched background rows fill the population.
    """

    if mode not in WARM_START_MODES:
        raise ValueError(f"mode must be one of {WARM_START_MODES}")
    if not str(candidate_source):
        raise ValueError("candidate_source cannot be empty")
    if not generated_field_names:
        raise ValueError("generated_field_names cannot be empty")
    if len(generated_field_names) != len(set(generated_field_names)):
        raise ValueError("generated_field_names contains duplicates")
    bounds = bounds or DesignSpaceBounds()
    background, param_names = build_background_population(
        bounds, population_size, background_seed, background_method
    )
    _, lower, upper = optimizer_parameter_space(bounds)
    unknown = set(generated_field_names).difference(param_names)
    if unknown:
        raise ValueError(
            f"generated fields are absent from 19D optimizer schema: {sorted(unknown)}"
        )

    if mode == "random":
        effective_ratio = 0.0
    elif mode == "generator":
        effective_ratio = 1.0
    else:
        if generator_ratio is None:
            raise ValueError("hybrid mode requires generator_ratio")
        effective_ratio = float(generator_ratio)
    if not 0.0 <= effective_ratio <= 1.0:
        raise ValueError("generator_ratio must be between zero and one")
    requested_generated_count = int(round(population_size * effective_ratio))

    candidate_matrix, candidate_metadata = _candidate_matrix(
        generated_candidates, generated_field_names
    )
    candidate_matrix, candidate_metadata, duplicate_rows = (
        _validate_and_deduplicate_candidates(
            candidate_matrix,
            candidate_metadata,
            generated_field_names,
            param_names,
            lower,
            upper,
            duplicate_tolerance,
        )
    )
    selected_count = min(requested_generated_count, len(candidate_matrix))
    output = background.copy()
    generated_indices = [param_names.index(name) for name in generated_field_names]
    if selected_count:
        output[:selected_count, generated_indices] = candidate_matrix[:selected_count]

    provenance_rows: List[Dict[str, Any]] = []
    for index in range(population_size):
        if index < selected_count:
            source = str(candidate_source)
            generated_metadata = candidate_metadata[index]
        elif index < requested_generated_count:
            source = "random_fill"
            generated_metadata = {}
        else:
            source = "random"
            generated_metadata = {}
        record: Dict[str, Any] = {
            "population_index": index,
            "source": source,
            "mode": mode,
            "background_row": index,
            "background_seed": int(background_seed),
            "background_method": background_method,
            "generated_fields": json.dumps(
                list(generated_field_names) if index < selected_count else []
            ),
        }
        record.update(generated_metadata)
        provenance_rows.append(record)

    digest = hashlib.sha256(output.astype("<f8", copy=False).tobytes()).hexdigest()
    metadata = {
        "mode": mode,
        "population_size": population_size,
        "generator_ratio": effective_ratio,
        "requested_generated_count": requested_generated_count,
        "selected_generated_count": selected_count,
        "random_fill_count": requested_generated_count - selected_count,
        "planned_random_count": population_size - requested_generated_count,
        "input_generated_count": len(candidate_matrix) + len(duplicate_rows),
        "unique_generated_count": len(candidate_matrix),
        "duplicate_generated_count": len(duplicate_rows),
        "duplicate_generated_source_rows": duplicate_rows,
        "truncated_generated_count": max(0, len(candidate_matrix) - selected_count),
        "generated_fields": list(generated_field_names),
        "candidate_source": str(candidate_source),
        "background_seed": int(background_seed),
        "background_method": background_method,
        "population_sha256_float64": digest,
    }
    population = InitialPopulation(
        genes=output,
        provenance=[dict(value) for value in provenance_rows],
        metadata=metadata,
        # Generator V1 fields are continuous.  Keeping the frozen 14D background
        # untouched makes Random and Hybrid backgrounds exactly comparable.
        apply_integer_projection=False,
    )
    return WarmStartBuildResult(
        population=population,
        provenance=pd.DataFrame(provenance_rows),
        metadata=metadata,
    )
