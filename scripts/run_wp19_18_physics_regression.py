#!/usr/bin/env python3
"""Run WP19-18 conservation and constraint-profile invariance regression."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics import Geometry19  # noqa: E402
from src.visualization import Geometry19DesignStore  # noqa: E402


DEFAULT_REGRESSION_ROOT = (
    REPO_ROOT / "results" / "visualization" / "wp19_16_geometry19_regression_20260825"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "physics" / "wp19_18_physics_regression_v1.json"
TOLERANCE_PATH = REPO_ROOT / "config" / "physics" / "geometry19_conservation_tolerances_v1.yaml"
OPERATING_POINT_POLICY_PATH = (
    REPO_ROOT
    / "config"
    / "physics"
    / "geometry19_operating_point_policy_reference_continuation_v1.yaml"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_tolerances() -> dict[str, Any]:
    payload = yaml.safe_load(TOLERANCE_PATH.read_text(encoding="utf-8"))
    if payload.get("profile_version") != TOLERANCE_PATH.stem:
        raise ValueError("Conservation tolerance filename/version mismatch")
    numeric = {
        name: float(payload[name])
        for name in (
            "combustor_energy_absolute_W",
            "shaft_power_absolute_W",
            "normalized_flow_absolute",
            "nozzle_velocity_absolute_m_s",
        )
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric.values()):
        raise ValueError("Conservation tolerances must be finite and positive")
    return {**payload, **numeric}


def load_operating_point_policy() -> dict[str, Any]:
    payload = yaml.safe_load(
        OPERATING_POINT_POLICY_PATH.read_text(encoding="utf-8")
    )
    if payload.get("policy_version") != OPERATING_POINT_POLICY_PATH.stem:
        raise ValueError("Operating-point policy filename/version mismatch")
    if payload.get("solver_policy") != "reference_continuation_high_flow_v1":
        raise ValueError("WP19-18 requires the frozen high-flow continuation policy")
    cases = payload.get("expected_boundary_no_flow_match_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Operating-point policy must list expected no-flow cases")
    return payload


def audit_conservation(
    physics: Mapping[str, Any], tolerances: Mapping[str, Any]
) -> dict[str, Any]:
    conservation = physics.get("conservation", {})
    operating = physics.get("operating_point", {})
    mass_flow_gas = abs(float(operating.get("mass_flow_gas_kg_s", math.nan)))
    mass_flow_limit = tolerances["normalized_flow_absolute"] * max(
        mass_flow_gas, 1.0e-12
    )
    definitions = {
        "combustor_energy_W": tolerances["combustor_energy_absolute_W"],
        "shaft_power_W": tolerances["shaft_power_absolute_W"],
        "mass_flow_kg_s": mass_flow_limit,
        "normalized_flow": tolerances["normalized_flow_absolute"],
        "nozzle_velocity_m_s": tolerances["nozzle_velocity_absolute_m_s"],
    }
    checks = {}
    for name, limit in definitions.items():
        value = float(conservation.get(name, math.nan))
        checks[name] = {
            "value": value,
            "absolute_value": abs(value),
            "absolute_tolerance": limit,
            "passed": math.isfinite(value) and abs(value) <= limit,
        }
    return {"checks": checks, "passed": all(item["passed"] for item in checks.values())}


def canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=True, sort_keys=True, separators=(",", ":"))


def profile_invariance(
    store: Geometry19DesignStore, design: Geometry19, quality: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    debug = store.regenerate(
        design,
        quality=quality,
        constraint_profile="pipeline_debug_v1",
    )
    engineering = store.regenerate(
        design,
        quality=quality,
        constraint_profile="engineering_candidate_v1",
    )
    checks = {
        "physics_identical": canonical_json(debug["physics"])
        == canonical_json(engineering["physics"]),
        "solver_identical": canonical_json(debug["solver"])
        == canonical_json(engineering["solver"]),
        "derived_geometry_identical": canonical_json(debug["derived_geometry"])
        == canonical_json(engineering["derived_geometry"]),
        "geometry_hash_identical": debug["geometry_hash"] == engineering["geometry_hash"],
        "lattice_hash_identical": debug["lattice_hash"] == engineering["lattice_hash"],
        "component_urls_identical": debug["component_urls"] == engineering["component_urls"],
        "constraint_profile_changed": debug["constraint_profile"]
        != engineering["constraint_profile"],
    }
    audit = {
        "checks": checks,
        "debug_accepted": debug["constraints"]["accepted"],
        "engineering_accepted": engineering["constraints"]["accepted"],
        "passed": all(checks.values()),
    }
    return audit, debug["physics"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = _read_json(args.regression_root / "summary.json")
    tolerances = load_tolerances()
    operating_point_policy = load_operating_point_policy()
    expected_no_flow_match_cases = set(
        operating_point_policy["expected_boundary_no_flow_match_cases"]
    )
    store = Geometry19DesignStore(REPO_ROOT, args.storage_root)
    records = []
    for case in source["cases"]:
        if case.get("status") == "expected_numerical_hard_failure":
            records.append(
                {
                    "case": case["case"],
                    "status": "expected_numerical_hard_failure",
                    "failure_code": case["expected_failure_rule"]["constraint"],
                    "passed": True,
                }
            )
            continue
        case_root = args.regression_root / "cases" / case["case"]
        design_mapping = _read_json(case_root / "geometry19.json")
        invariance, physics = profile_invariance(
            store, Geometry19.from_mapping(design_mapping), args.quality
        )
        converged = bool(physics["solver"]["converged"])
        conservation = audit_conservation(physics, tolerances) if converged else None
        failure_code = physics["solver"].get("failure_code")
        expected_no_flow_match = (
            case["case"] in expected_no_flow_match_cases
            and failure_code == "NO_FLOW_MATCH"
        )
        record_passed = invariance["passed"] and (
            (converged and conservation["passed"]) or expected_no_flow_match
        )
        if converged:
            status = "converged"
        elif expected_no_flow_match:
            status = "expected_no_flow_match"
        else:
            status = "solver_failed"
        records.append(
            {
                "case": case["case"],
                "status": status,
                "failure_code": failure_code,
                "solver_message": physics["solver"].get("message", ""),
                "flow_root_diagnostics": {
                    name: physics.get("operating_point", {}).get(name)
                    for name in (
                        "mass_flow_search_min_kg_s",
                        "mass_flow_search_max_kg_s",
                        "inlet_choked_capacity_kg_s",
                        "flow_root_count",
                        "flow_root_mass_flow_air_kg_s",
                        "flow_root_normalized_residuals",
                    )
                },
                "profile_invariance": invariance,
                "conservation": conservation,
                "passed": record_passed,
            }
        )
    failure_distribution: dict[str, int] = {}
    for record in records:
        if record["status"] == "solver_failed":
            code = str(record["failure_code"])
            failure_distribution[code] = failure_distribution.get(code, 0) + 1
    result = {
        "work_package": "WP19-18",
        "source_work_package": source["work_package"],
        "constraint_profile_pair": [
            "pipeline_debug_v1",
            "engineering_candidate_v1",
        ],
        "conservation_tolerance_profile": tolerances["profile_version"],
        "conservation_tolerances": tolerances,
        "operating_point_policy": operating_point_policy["solver_policy"],
        "operating_point_policy_version": operating_point_policy["policy_version"],
        "case_count": len(records),
        "converged_count": sum(record["status"] == "converged" for record in records),
        "expected_hard_failure_count": sum(
            record["status"] == "expected_numerical_hard_failure"
            for record in records
        ),
        "expected_no_flow_match_count": sum(
            record["status"] == "expected_no_flow_match" for record in records
        ),
        "selected_multiple_root_count": sum(
            record.get("flow_root_diagnostics", {}).get("flow_root_count", 0) > 1
            and record["status"] == "converged"
            for record in records
        ),
        "solver_failure_count": sum(record["status"] == "solver_failed" for record in records),
        "failure_code_distribution": failure_distribution,
        "profile_invariance_passed_count": sum(
            record.get("profile_invariance", {}).get("passed", False)
            for record in records
        ),
        "conservation_passed_count": sum(
            record.get("conservation", {}).get("passed", False)
            for record in records
            if record.get("conservation") is not None
        ),
        "passed_count": sum(record["passed"] for record in records),
        "failed_cases": [record["case"] for record in records if not record["passed"]],
        "passed": all(record["passed"] for record in records),
        "cases": records,
    }
    _write_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", type=Path, default=DEFAULT_REGRESSION_ROOT)
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_REGRESSION_ROOT / "_design_store_v3",
    )
    parser.add_argument("--quality", choices=("preview_mesh", "export_mesh"), default="preview_mesh")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(
        f"WP19-18: {result['passed_count']}/{result['case_count']} passed; "
        f"converged={result['converged_count']}; solver_failures={result['solver_failure_count']}"
    )
    if result["failure_code_distribution"]:
        print(json.dumps(result["failure_code_distribution"], sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
