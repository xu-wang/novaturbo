"""WP19-23F lossless performance comparison tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_delivery import (
    Geometry19DeliveryError,
    performance_matches,
    run_delivery_stage,
    validate_delivery_source_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "config/inverse_design/targets/wp19_23_smoke_test0_v1.yaml"


def test_performance_match_requires_all_three_axes() -> None:
    reference = {"thrust_N": 10.0, "tsfc_kg_N_s": 1.0e-4, "mass_kg": 2.0}
    assert performance_matches(reference, dict(reference))
    assert not performance_matches(reference, {**reference, "mass_kg": 2.1})


def test_performance_match_uses_tight_relative_tolerance() -> None:
    reference = {"thrust_N": 10.0, "tsfc_kg_N_s": 1.0e-4, "mass_kg": 2.0}
    recalculated = {"thrust_N": 10.0 + 1.0e-12, "tsfc_kg_N_s": 1.0e-4, "mass_kg": 2.0}
    assert performance_matches(reference, recalculated)
    assert not performance_matches(reference, recalculated, relative_tolerance=0.0)


@pytest.mark.parametrize("stage", ("WP19-23C", "WP19-23D", "WP19-23E"))
def test_delivery_accepts_any_terminal_inverse_stage(stage: str) -> None:
    assert validate_delivery_source_summary(
        {"work_package": stage, "target_id": "target", "downstream_stop": True},
        "target",
    ) == stage


def test_delivery_rejects_nonterminal_or_unknown_source() -> None:
    with pytest.raises(Geometry19DeliveryError, match="not ready"):
        validate_delivery_source_summary(
            {"work_package": "WP19-23C", "target_id": "target", "downstream_stop": False},
            "target",
        )
    with pytest.raises(Geometry19DeliveryError, match="must be WP19-23C"):
        validate_delivery_source_summary(
            {"work_package": "WP19-23B", "target_id": "target", "downstream_stop": True},
            "target",
        )


def test_delivery_rejects_shadow_context_before_creating_outputs(tmp_path: Path) -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_variable_thermo_shadow"
    )
    output = tmp_path / "delivery"

    with pytest.raises(Geometry19DeliveryError, match="disabled"):
        run_delivery_stage(
            REPO_ROOT,
            TARGET_PATH,
            source_root=tmp_path / "source",
            output_root=output,
            pipeline_context=context,
        )

    assert not output.exists()
