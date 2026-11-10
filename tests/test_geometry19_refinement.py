"""WP19-23D deterministic seed-selection tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.ai.geometry19_refinement import select_refinement_seeds


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fields() -> list[dict]:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(encoding="utf-8")
    )
    return list(schema["design_variables"])


def _geometry(fields: list[dict], offset: float) -> dict:
    result = {}
    for item in fields:
        lower, upper = item["bounds"]
        value = item["reference_value"]
        if item["dtype"] != "int64":
            value = min(upper, max(lower, value + offset * (upper - lower)))
        result[item["name"]] = value
    return result


def _row(fields: list[dict], identifier: str, error: float, offset: float, root: str) -> dict:
    return {
        "evaluation_id": identifier,
        "candidate_id": int(identifier.split(":")[-1]),
        "design_hash": identifier,
        "geometry19": _geometry(fields, offset),
        "physics_acceptable": True,
        "total_true_error": error,
        "root_classification": root,
    }


def test_seed_selection_deduplicates_and_prioritizes_unique_roots() -> None:
    fields = _fields()
    first = _row(fields, "direct:1", 0.2, 0.0, "CONVERGED_UNIQUE")
    duplicate = dict(first, evaluation_id="dflow:1", total_true_error=0.1)
    multiple = _row(
        fields, "dflow:2", 0.01, 0.2, "CONVERGED_SELECTED_MULTIPLE"
    )
    unique_far = _row(fields, "direct:3", 0.3, -0.2, "CONVERGED_UNIQUE")
    invalid = dict(
        _row(fields, "direct:4", 0.001, 0.1, "CONVERGED_UNIQUE"),
        physics_acceptable=False,
    )
    selected = select_refinement_seeds(
        [first, duplicate, multiple, unique_far, invalid], fields, 3
    )
    assert [row["evaluation_id"] for row in selected] == ["dflow:1", "direct:3", "dflow:2"]
    assert all(len(row["normalized_geometry"]) == 19 for row in selected)


def test_seed_selection_respects_maximum() -> None:
    fields = _fields()
    rows = [
        _row(fields, f"direct:{index}", 0.1 + index, index * 0.01, "CONVERGED_UNIQUE")
        for index in range(10)
    ]
    assert len(select_refinement_seeds(rows, fields, 8)) == 8

