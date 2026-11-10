"""WP19-21 campaign allocation, streaming merge, and duplicate tests."""

import json

import pytest

from src.physics.wp19_21_expansion import (
    WP1921ExpansionError,
    allocate_preview_budgets,
    merge_campaign_datasets,
)


def _record(sample_id, value, converged=True, preview=False):
    return {
        "sample_id": sample_id,
        "x_19": {"x": value},
        "solver_status": {"converged": converged, "classification": "OK" if converged else "NO_FLOW_MATCH"},
        "mesh_preview_reference": {"requested": preview, "available": preview},
        "provenance": {},
    }


def _write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_preview_budget_is_exact_and_balanced():
    budgets = allocate_preview_budgets(256, 5)
    assert budgets == [52, 51, 51, 51, 51]
    assert sum(budgets) == 256


def test_streaming_merge_exports_regression_and_feasibility_sets(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _write(first, [_record(0, 1, True, True), _record(1, 2, False)])
    _write(second, [_record(0, 3, True), _record(1, 4, False)])
    summary = merge_campaign_datasets(
        [
            {"campaign_id": "a", "seed": 1, "dataset_jsonl": first},
            {"campaign_id": "b", "seed": 2, "dataset_jsonl": second},
        ],
        tmp_path / "merged",
        4,
    )
    assert summary["sample_count"] == summary["unique_design_count"] == 4
    assert summary["converged_count"] == 2
    assert len((tmp_path / "merged/regression_converged.jsonl").read_text().splitlines()) == 2
    assert len((tmp_path / "merged/feasibility_all.jsonl").read_text().splitlines()) == 4


def test_streaming_merge_rejects_duplicate_x19(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _write(first, [_record(0, 1)])
    _write(second, [_record(0, 1)])
    with pytest.raises(WP1921ExpansionError, match="Duplicate x_19"):
        merge_campaign_datasets(
            [
                {"campaign_id": "a", "seed": 1, "dataset_jsonl": first},
                {"campaign_id": "b", "seed": 2, "dataset_jsonl": second},
            ],
            tmp_path / "merged",
            2,
        )
