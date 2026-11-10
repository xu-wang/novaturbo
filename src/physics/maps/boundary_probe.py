"""Reusable, evaluation-backed probes for two-coordinate map boundaries."""

from __future__ import annotations

from typing import Any, Mapping

from src.physics.maps.base import ComponentMap


def probe_axis_boundaries(
    component_map: ComponentMap,
    *,
    fixed_coordinates: Mapping[str, float],
    axis_name: str,
    center_value: float,
    minimum_margin_name: str,
    maximum_margin_name: str,
    fraction: float,
) -> dict[str, Any]:
    """Evaluate four signed probes around interpolated operating boundaries."""

    if axis_name != component_map.coordinate_names[1]:
        raise ValueError("Boundary probe axis must be the map's second coordinate")
    if set(fixed_coordinates) != {component_map.coordinate_names[0]}:
        raise ValueError("Boundary probe must fix exactly the map's first coordinate")
    if not 0.0 < fraction < 0.1:
        raise ValueError("Boundary probe fraction must be in (0, 0.1)")
    center = component_map.evaluate(
        {**fixed_coordinates, axis_name: float(center_value)}
    )
    minimum = float(center.margins[minimum_margin_name])
    maximum = float(center.margins[maximum_margin_name])
    span = maximum - minimum
    if span <= 0.0:
        raise ValueError("Map boundary span must be positive")
    probes = {
        "below_minimum": minimum - fraction * span,
        "inside_near_minimum": minimum + fraction * span,
        "inside_near_maximum": maximum - fraction * span,
        "above_maximum": maximum + fraction * span,
    }
    evaluated = {
        name: component_map.evaluate(
            {**fixed_coordinates, axis_name: float(value)}
        )
        for name, value in probes.items()
    }
    signed_margins = {
        "below_minimum": probes["below_minimum"]
        - float(evaluated["below_minimum"].margins[minimum_margin_name]),
        "inside_near_minimum": probes["inside_near_minimum"]
        - float(evaluated["inside_near_minimum"].margins[minimum_margin_name]),
        "inside_near_maximum": float(
            evaluated["inside_near_maximum"].margins[maximum_margin_name]
        )
        - probes["inside_near_maximum"],
        "above_maximum": float(
            evaluated["above_maximum"].margins[maximum_margin_name]
        )
        - probes["above_maximum"],
    }
    passed = (
        signed_margins["below_minimum"] < 0.0
        and signed_margins["inside_near_minimum"] > 0.0
        and signed_margins["inside_near_maximum"] > 0.0
        and signed_margins["above_maximum"] < 0.0
    )
    return {
        "minimum_boundary": minimum,
        "maximum_boundary": maximum,
        "probe_values": probes,
        "signed_margins": signed_margins,
        "map_in_domain": {
            name: result.in_domain for name, result in evaluated.items()
        },
        "pass": passed,
    }
