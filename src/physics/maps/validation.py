"""Map-package validation gates independent of an engine pipeline run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.physics.maps.base import ComponentMap, load_component_map
from src.physics.maps.hecc import load_hecc_points, validate_hecc_against_measurements


def validate_component_map_package(package_path: str | Path) -> dict[str, Any]:
    """Return a serializable structural and evidence report for one map package."""

    package = Path(package_path).resolve()
    component_map = load_component_map(package)
    checks: dict[str, bool] = {
        "coordinate_axes_strictly_increasing": all(
            bool(np.all(np.diff(component_map.coordinates[name]) > 0.0))
            for name in component_map.coordinate_names
        ),
        "outputs_finite": all(
            bool(np.all(np.isfinite(surface)))
            for surface in component_map.outputs.values()
        ),
        "boundaries_finite": all(
            bool(np.all(np.isfinite(line)))
            for line in component_map.boundaries.values()
        ),
    }
    if component_map.component_type == "compressor":
        surge = component_map.boundaries.get("surge_flow_coefficient_ratio")
        choke = component_map.boundaries.get("choke_flow_coefficient_ratio")
        efficiency = component_map.outputs.get("total_to_total_efficiency")
        checks.update(
            {
                "surge_boundary_below_choke_boundary": bool(
                    surge is not None and choke is not None and np.all(surge < choke)
                ),
                "efficiency_strictly_between_zero_and_one": bool(
                    efficiency is not None
                    and np.all(efficiency > 0.0)
                    and np.all(efficiency < 1.0)
                ),
                "positive_work_closure": bool(
                    any(
                        name in component_map.outputs
                        and np.all(component_map.outputs[name] > 0.0)
                        for name in ("work_coefficient", "euler_work_factor")
                    )
                ),
            }
        )
    elif component_map.component_type == "turbine":
        minimum = component_map.boundaries.get(
            "minimum_corrected_mass_flux_ratio"
        )
        maximum = component_map.boundaries.get(
            "maximum_corrected_mass_flux_ratio"
        )
        efficiency = component_map.outputs.get("total_to_total_efficiency")
        checks.update(
            {
                "minimum_capacity_below_maximum_capacity": bool(
                    minimum is not None
                    and maximum is not None
                    and np.all(minimum < maximum)
                ),
                "efficiency_strictly_between_zero_and_one": bool(
                    efficiency is not None
                    and np.all(efficiency > 0.0)
                    and np.all(efficiency < 1.0)
                ),
            }
        )
    elif component_map.component_type == "inlet":
        minimum = component_map.boundaries.get("minimum_capture_area_ratio")
        maximum = component_map.boundaries.get("maximum_capture_area_ratio")
        recovery = component_map.outputs.get("total_pressure_recovery")
        discharge = component_map.outputs.get("discharge_coefficient")
        checks.update(
            {
                "minimum_capture_area_below_maximum": bool(
                    minimum is not None
                    and maximum is not None
                    and np.all(minimum < maximum)
                ),
                "pressure_recovery_in_zero_one": bool(
                    recovery is not None
                    and np.all(recovery > 0.0)
                    and np.all(recovery <= 1.0)
                ),
                "discharge_coefficient_in_zero_one": bool(
                    discharge is not None
                    and np.all(discharge > 0.0)
                    and np.all(discharge <= 1.0)
                ),
            }
        )
    elif component_map.component_type == "combustor":
        lean = component_map.boundaries.get("minimum_equivalence_ratio")
        rich = component_map.boundaries.get("maximum_equivalence_ratio")
        efficiency = component_map.outputs.get("combustion_efficiency")
        pressure_loss = component_map.outputs.get("total_pressure_loss_fraction")
        checks.update(
            {
                "lean_boundary_below_rich_boundary": bool(
                    lean is not None and rich is not None and np.all(lean < rich)
                ),
                "combustion_efficiency_in_zero_one": bool(
                    efficiency is not None
                    and np.all(efficiency > 0.0)
                    and np.all(efficiency <= 1.0)
                ),
                "pressure_loss_fraction_in_zero_one": bool(
                    pressure_loss is not None
                    and np.all(pressure_loss >= 0.0)
                    and np.all(pressure_loss < 1.0)
                ),
            }
        )
    elif component_map.component_type == "nozzle":
        minimum = component_map.boundaries.get("minimum_reynolds_ratio")
        maximum = component_map.boundaries.get("maximum_reynolds_ratio")
        recovery = component_map.outputs.get("total_pressure_recovery")
        discharge = component_map.outputs.get("discharge_coefficient")
        checks.update(
            {
                "minimum_reynolds_below_maximum": bool(
                    minimum is not None
                    and maximum is not None
                    and np.all(minimum < maximum)
                ),
                "pressure_recovery_in_zero_one": bool(
                    recovery is not None
                    and np.all(recovery > 0.0)
                    and np.all(recovery <= 1.0)
                ),
                "discharge_coefficient_in_zero_one": bool(
                    discharge is not None
                    and np.all(discharge > 0.0)
                    and np.all(discharge <= 1.0)
                ),
            }
        )
    evidence: dict[str, Any] | None = None
    if component_map.map_id == "nasa_hecc_post_test_cfd_v1":
        evidence = validate_hecc_against_measurements(
            component_map, load_hecc_points(package / "source_points.csv")
        )
    structural_pass = all(checks.values())
    shadow_ready = structural_pass and (
        evidence is None or bool(evidence["shadow_ready"])
    )
    return {
        "map_id": component_map.map_id,
        "map_version": component_map.map_version,
        "source_hash": component_map.source_hash,
        "formal_use_allowed": component_map.formal_use_allowed,
        "checks": checks,
        "structural_pass": structural_pass,
        "evidence_validation": evidence,
        "shadow_ready": shadow_ready,
        "engineering_ready": bool(
            structural_pass
            and evidence is not None
            and evidence["engineering_ready"]
            and component_map.formal_use_allowed
        ),
    }
