"""WP19-20 sensitivity helper tests."""

from pathlib import Path

import yaml
import pytest

from src.physics.wp19_20_sensitivity import build_positive_oat_cases, spearman_sensitivity


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_positive_oat_matrix_covers_all_19_coordinates_once():
    schema = yaml.safe_load((REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text())
    cases = build_positive_oat_cases(schema)
    assert len(cases) == 20
    assert cases[0][0] is None
    assert [variable["name"] for variable, _ in cases[1:]] == list(cases[0][1].coordinate_names())
    for variable, design in cases[1:]:
        assert getattr(design, variable["name"]) > variable["reference_value"]


def test_rank_sensitivity_reports_active_channels():
    records = []
    for value in range(5):
        records.append({
            "geometry19": {"x": value},
            "physics": {
                "solver": {"converged": True},
                "performance": {"thrust_N": value * 2, "tsfc_kg_N_s": 5 - value, "mass_kg": value + 1},
                "constraints": {"g": value - 2},
            },
        })
    audit = spearman_sensitivity(records, ["x"])
    assert audit["converged_sample_count"] == 5
    assert audit["variables"]["x"]["nonconstant_channel_count"] == 4
    assert audit["variables"]["x"]["max_abs_spearman"] == pytest.approx(1.0)
