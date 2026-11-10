"""Deterministic conversion and validation helpers for the NASA HECC map."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from src.physics.maps.base import ComponentMap


STANDARD_TEMPERATURE_K = 288.15
STANDARD_CP_J_KG_K = 1004.5
STANDARD_GAMMA = 1.4
DESIGN_SPEED_RPM = 21_789.0
IMPELLER_EXIT_RADIUS_M = 8.497 * 0.0254
DESIGN_CORRECTED_FLOW_LBM_S = 10.862
FLOW_GRID = np.asarray((0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.11))


@dataclass(frozen=True)
class HECCPoint:
    dataset: str
    speed_ratio: float
    corrected_flow_lbm_s: float
    total_pressure_ratio: float
    tt_adiabatic_efficiency: float

    @property
    def flow_coefficient_ratio(self) -> float:
        return self.corrected_flow_lbm_s / (
            DESIGN_CORRECTED_FLOW_LBM_S * self.speed_ratio
        )

    @property
    def work_coefficient(self) -> float:
        ideal_work = STANDARD_CP_J_KG_K * STANDARD_TEMPERATURE_K * (
            self.total_pressure_ratio ** ((STANDARD_GAMMA - 1.0) / STANDARD_GAMMA)
            - 1.0
        )
        actual_work = ideal_work / self.tt_adiabatic_efficiency
        angular_speed = 2.0 * math.pi * DESIGN_SPEED_RPM * self.speed_ratio / 60.0
        tip_speed = angular_speed * IMPELLER_EXIT_RADIUS_M
        return actual_work / tip_speed**2


def load_hecc_points(path: str | Path) -> tuple[HECCPoint, ...]:
    """Load the hand-transcribed report tables under an exact column contract."""

    expected = (
        "dataset",
        "speed_ratio",
        "corrected_flow_lbm_s",
        "total_pressure_ratio",
        "tt_adiabatic_efficiency",
    )
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"HECC source columns must be exactly {expected}")
        points = tuple(
            HECCPoint(
                dataset=str(row["dataset"]),
                speed_ratio=float(row["speed_ratio"]),
                corrected_flow_lbm_s=float(row["corrected_flow_lbm_s"]),
                total_pressure_ratio=float(row["total_pressure_ratio"]),
                tt_adiabatic_efficiency=float(row["tt_adiabatic_efficiency"]),
            )
            for row in reader
        )
    if not points:
        raise ValueError("HECC source table is empty")
    if any(
        point.dataset not in {"post_test_cfd", "measured_100_percent"}
        or not 0.0 < point.speed_ratio <= 2.0
        or point.corrected_flow_lbm_s <= 0.0
        or point.total_pressure_ratio <= 1.0
        or not 0.0 < point.tt_adiabatic_efficiency < 1.0
        for point in points
    ):
        raise ValueError("HECC source table contains a nonphysical row")
    return points


def build_hecc_map_data(points: Iterable[HECCPoint]) -> dict[str, object]:
    """Rectify the five CFD speedlines; clamp only outside each measured line."""

    selected = tuple(point for point in points if point.dataset == "post_test_cfd")
    speeds = np.asarray(sorted({point.speed_ratio for point in selected}))
    if not np.array_equal(speeds, np.asarray((0.85, 0.90, 0.95, 1.00, 1.05))):
        raise ValueError("HECC CFD source must contain the locked five speedlines")
    work_surface: list[list[float]] = []
    efficiency_surface: list[list[float]] = []
    surge: list[float] = []
    choke: list[float] = []
    for speed in speeds:
        line = sorted(
            (point for point in selected if point.speed_ratio == speed),
            key=lambda point: point.flow_coefficient_ratio,
        )
        flow = np.asarray([point.flow_coefficient_ratio for point in line])
        work = np.asarray([point.work_coefficient for point in line])
        efficiency = np.asarray([point.tt_adiabatic_efficiency for point in line])
        if len(line) < 3 or np.any(np.diff(flow) <= 0.0):
            raise ValueError(f"HECC speedline {speed} is incomplete or unordered")
        surge.append(float(flow[0]))
        choke.append(float(flow[-1]))
        work_surface.append(np.interp(FLOW_GRID, flow, work).tolist())
        efficiency_surface.append(np.interp(FLOW_GRID, flow, efficiency).tolist())
    return {
        "coordinates": {
            "corrected_speed_ratio": speeds.tolist(),
            "flow_coefficient_ratio": FLOW_GRID.tolist(),
        },
        "outputs": {
            "work_coefficient": work_surface,
            "total_to_total_efficiency": efficiency_surface,
        },
        "boundaries": {
            "surge_flow_coefficient_ratio": surge,
            "choke_flow_coefficient_ratio": choke,
        },
    }


def validate_hecc_against_measurements(
    component_map: ComponentMap, points: Iterable[HECCPoint]
) -> dict[str, object]:
    """Compare the CFD-derived map with the independent 100% experimental line."""

    measured_table = tuple(
        point for point in points if point.dataset == "measured_100_percent"
    )
    if len(measured_table) != 20:
        raise ValueError("Locked HECC measured validation line must contain 20 points")
    # Table A.4(a) is an inlet-suppression sweep, not a conventional speedline.
    # Only the explicitly reported 10.862 lbm/s design condition is a like-for-like
    # validation anchor for this conventional map.
    measured = tuple(
        point
        for point in measured_table
        if math.isclose(point.corrected_flow_lbm_s, DESIGN_CORRECTED_FLOW_LBM_S)
    )
    if len(measured) != 1:
        raise ValueError("HECC measured design-point anchor is missing or ambiguous")
    pressure_errors = []
    efficiency_errors = []
    work_errors = []
    for point in measured:
        result = component_map.evaluate(
            {
                "corrected_speed_ratio": point.speed_ratio,
                "flow_coefficient_ratio": point.flow_coefficient_ratio,
            }
        )
        predicted_efficiency = float(result.values["total_to_total_efficiency"])
        predicted_work = float(result.values["work_coefficient"])
        tip_speed = (
            2.0
            * math.pi
            * DESIGN_SPEED_RPM
            * point.speed_ratio
            / 60.0
            * IMPELLER_EXIT_RADIUS_M
        )
        ideal_work = predicted_work * tip_speed**2 * predicted_efficiency
        predicted_pressure_ratio = (
            1.0 + ideal_work / (STANDARD_CP_J_KG_K * STANDARD_TEMPERATURE_K)
        ) ** (STANDARD_GAMMA / (STANDARD_GAMMA - 1.0))
        pressure_errors.append(
            (predicted_pressure_ratio - point.total_pressure_ratio)
            / point.total_pressure_ratio
        )
        efficiency_errors.append(
            predicted_efficiency - point.tt_adiabatic_efficiency
        )
        work_errors.append(predicted_work - point.work_coefficient)

    def metrics(errors: list[float]) -> dict[str, float]:
        values = np.asarray(errors)
        return {
            "mean_error": float(np.mean(values)),
            "mean_absolute_error": float(np.mean(np.abs(values))),
            "root_mean_square_error": float(np.sqrt(np.mean(values**2))),
            "max_absolute_error": float(np.max(np.abs(values))),
        }

    comparison = {
        "pressure_ratio_relative": metrics(pressure_errors),
        "efficiency_absolute": metrics(efficiency_errors),
        "work_coefficient_absolute": metrics(work_errors),
    }
    shadow_ready = (
        comparison["pressure_ratio_relative"]["max_absolute_error"] <= 0.12
        and comparison["efficiency_absolute"]["max_absolute_error"] <= 0.10
        and comparison["work_coefficient_absolute"]["max_absolute_error"] <= 0.12
    )
    return {
        "validation_dataset": "NASA/CR-2014-218114/REV1 Table A.4(a) design point",
        "validation_speedline_count": 0,
        "validation_point_count": len(measured),
        "metrics": comparison,
        "shadow_tolerances": {
            "pressure_ratio_relative_max_abs": 0.12,
            "efficiency_absolute_max_abs": 0.10,
            "work_coefficient_absolute_max_abs": 0.12,
        },
        "shadow_ready": shadow_ready,
        "engineering_ready": False,
        "engineering_blockers": [
            "only_one_independent_measured_design_point",
            "NASA_HECC_archive_object_returns_HTTP_403_as_of_2026-09-10",
            "cross_geometry_similarity_not_validated",
        ],
    }
