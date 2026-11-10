"""Deterministic full-19D reference-direction selection for warm starts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .dataset import evaluate_design


def objective_scaling(
    reference: pd.DataFrame,
    objective_names: Sequence[str],
    maximize: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(objective_names) != len(maximize):
        raise ValueError("objective_names and maximize lengths differ")
    if "is_valid" in reference:
        reference = reference.loc[reference["is_valid"].astype(bool)]
    columns = []
    for name, is_maximized in zip(objective_names, maximize):
        if name not in reference:
            raise ValueError(f"Preference reference dataset is missing {name}")
        values = reference[name].to_numpy(dtype=float)
        columns.append(-values if is_maximized else values)
    matrix = np.column_stack(columns)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if not len(matrix):
        raise ValueError("Preference reference dataset has no finite valid rows")
    return np.min(matrix, axis=0), np.max(matrix, axis=0)


def evaluate_preference_pool(
    genes: np.ndarray,
    param_names: Sequence[str],
    objective_names: Sequence[str],
    maximize: Sequence[bool],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    genes = np.asarray(genes, dtype=float)
    if genes.ndim != 2 or genes.shape[1] != len(param_names):
        raise ValueError("Preference genes do not match optimizer parameter schema")
    rows: List[Dict[str, Any]] = []
    objectives = []
    feasible = []
    for index, gene_row in enumerate(genes):
        performance = evaluate_design(pd.Series(dict(zip(param_names, gene_row))))
        is_feasible = bool(performance.get("is_valid", False))
        internal = []
        for name, is_maximized in zip(objective_names, maximize):
            value = float(performance.get(name, np.nan))
            internal.append(-value if is_maximized else value)
        finite = bool(np.isfinite(internal).all())
        is_feasible = is_feasible and finite
        if not is_feasible:
            internal = [1.0e10] * len(objective_names)
        row: Dict[str, Any] = {
            "pool_index": index,
            "is_feasible": is_feasible,
        }
        row.update({f"objective_internal_{i}": value for i, value in enumerate(internal)})
        row.update({f"performance__{key}": value for key, value in performance.items()})
        rows.append(row)
        objectives.append(internal)
        feasible.append(is_feasible)
    return (
        pd.DataFrame(rows),
        np.asarray(objectives, dtype=float),
        np.asarray(feasible, dtype=bool),
    )


def select_reference_directions(
    objectives: np.ndarray,
    feasible: np.ndarray,
    n_select: int,
    directions: Sequence[Sequence[float]],
    objective_min: np.ndarray,
    objective_max: np.ndarray,
) -> Tuple[List[int], pd.DataFrame]:
    """Round-robin achievement-scalarizing selection over fixed directions."""

    objectives = np.asarray(objectives, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    if objectives.ndim != 2 or feasible.shape != (len(objectives),):
        raise ValueError("Preference objective/feasible shapes are incompatible")
    if n_select < 0:
        raise ValueError("n_select must be non-negative")
    direction_matrix = np.asarray(directions, dtype=float)
    if direction_matrix.ndim != 2 or direction_matrix.shape[1] != objectives.shape[1]:
        raise ValueError("Reference directions do not match objective dimension")
    if np.any(direction_matrix <= 0.0) or not np.isfinite(direction_matrix).all():
        raise ValueError("Reference directions must be finite and strictly positive")
    direction_matrix = direction_matrix / direction_matrix.sum(axis=1, keepdims=True)
    scale = np.maximum(
        np.asarray(objective_max, dtype=float) - np.asarray(objective_min, dtype=float),
        1.0e-12,
    )
    normalized = (objectives - np.asarray(objective_min, dtype=float)) / scale
    eligible = np.flatnonzero(feasible & np.all(np.isfinite(normalized), axis=1))
    target_count = min(int(n_select), len(eligible))
    if target_count == 0:
        return [], pd.DataFrame(
            columns=["selection_rank", "pool_index", "direction_index", "asf_score"]
        )
    scores = np.full((len(direction_matrix), len(objectives)), np.inf, dtype=float)
    for direction_index, direction in enumerate(direction_matrix):
        scores[direction_index, eligible] = np.max(
            normalized[eligible] / np.maximum(direction, 1.0e-12), axis=1
        )
    ordered = [
        eligible[np.argsort(scores[index, eligible], kind="stable")]
        for index in range(len(direction_matrix))
    ]
    cursors = [0] * len(ordered)
    selected: List[int] = []
    selected_set = set()
    records: List[Dict[str, Any]] = []
    while len(selected) < target_count:
        progress = False
        for direction_index, candidates in enumerate(ordered):
            while (
                cursors[direction_index] < len(candidates)
                and int(candidates[cursors[direction_index]]) in selected_set
            ):
                cursors[direction_index] += 1
            if cursors[direction_index] >= len(candidates):
                continue
            pool_index = int(candidates[cursors[direction_index]])
            cursors[direction_index] += 1
            selected.append(pool_index)
            selected_set.add(pool_index)
            records.append(
                {
                    "selection_rank": len(selected),
                    "pool_index": pool_index,
                    "direction_index": direction_index,
                    "asf_score": float(scores[direction_index, pool_index]),
                }
            )
            progress = True
            if len(selected) >= target_count:
                break
        if not progress:
            break
    return selected, pd.DataFrame(records)
