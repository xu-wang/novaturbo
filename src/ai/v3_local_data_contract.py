"""Freeze center-disjoint assignments for future V3 local development data."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from src.ai.v3_local_center_audit import verify_v3_local_center_audit
from src.ai.v3_local_pilot import verify_v3_local_pilot_labels


CONFIG = "config/experiments/geometry22_v3_local_data_contract_v1.yaml"
CODE = "src/ai/v3_local_data_contract.py"
SPLITS = ("development", "validation", "holdout")


class V3LocalDataContractError(ValueError):
    """The frozen source or center assignment is incompatible."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (
        config.get("version") != "geometry22_v3_local_data_contract_v1"
        or config.get("scope") != "future_software_development_acquisition_only"
        or config.get("stratum") != "l0_v3_flow_root_count_numerical_branch_count_only"
        or config.get("assignment") != "sha256_seed_colon_center_design_id_sort_within_stratum"
        or config.get("inheritance") != "all_future_samples_from_center_keep_center_split"
        or config.get("split_before_future_labels") is not True
        or config.get("holdout_adaptive_top_up_allowed") is not False
        or config.get("future_label_qualification") != {
            "regression_candidate_requires": "solver_converged_and_finite_performance_and_borrowed_diagnostic_domain",
            "numerical_failure": "exclude_from_regression_keep_failure_diagnostic",
            "outside_borrowed_diagnostic_domain": "exclude_from_regression_keep_domain_diagnostic",
            "engineering_soft_constraints": "report_separately_do_not_use_as_regression_filter",
            "engine_feasible": "report_separately_no_software_delivery_claim",
            "train_centers": "development_only",
            "validation_centers": "validation_only",
            "holdout_centers": "holdout_only_no_adaptive_selection",
        }
        or config.get("v2_2_labels_reused") is not False
        or config.get("l0_l1_training_allowed") is not False
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
        or not isinstance(config.get("split_seed"), str)
        or not config["split_seed"]
        or set(config.get("stratum_split_counts", {})) != {1, 2}
        or any(set(counts) != set(SPLITS) or any(type(n) is not int or n < 0 for n in counts.values())
               for counts in config["stratum_split_counts"].values())
    ):
        raise V3LocalDataContractError("V3 local data contract config changed scope")
    return config


def _expected(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = _config(root)
    center_source = root / config["source_centers"]["path"]
    pilot_source = root / config["source_pilot"]["path"]
    if (
        _sha(center_source / "complete_manifest.json") != config["source_centers"]["manifest_sha256"]
        or _sha(center_source / "centers.jsonl") != config["source_centers"]["centers_sha256"]
        or _sha(pilot_source / "complete_manifest.json") != config["source_pilot"]["complete_manifest_sha256"]
        or _sha(pilot_source / "labels.jsonl") != config["source_pilot"]["labels_sha256"]
    ):
        raise V3LocalDataContractError("L0/L1 frozen source hash changed")
    verify_v3_local_center_audit(root, center_source)
    verify_v3_local_pilot_labels(root, pilot_source)
    centers = _rows(center_source / "centers.jsonl")
    pilot = _rows(pilot_source / "coordinates.jsonl")
    if (
        len(centers) != config["source_centers"]["expected_centers"]
        or len({row["design_id"] for row in centers}) != len(centers)
        or {row["center_design_id"] for row in pilot} != {row["design_id"] for row in centers}
        or {row["split"] for row in pilot} != set(SPLITS)
        or any(not row["candidate_for_local_pilot"] or row["engine_feasible"]
               or row["training_allowed"] for row in centers)
        or any(row["training_allowed"] for row in pilot)
    ):
        raise V3LocalDataContractError("L0/L1 center population or scope changed")
    rows = []
    for stratum, counts in sorted(config["stratum_split_counts"].items()):
        members = [row for row in centers if row["flow_root_count"] == stratum]
        if len(members) != sum(counts.values()):
            raise V3LocalDataContractError("L0 numerical branch-count stratum changed")
        members.sort(key=lambda row: (
            hashlib.sha256(f"{config['split_seed']}:{row['design_id']}".encode()).hexdigest(),
            row["design_id"],
        ))
        assignments = [split for split in SPLITS for _ in range(counts[split])]
        for center, split in zip(members, assignments, strict=True):
            rows.append({
                "center_design_id": center["design_id"],
                "source_v2_2_root_seed_id_provenance_only": center["source_v2_2_root_seed_id"],
                "l0_flow_root_count_numerical_only": stratum,
                "center_split": split,
                "integer_state": {
                    name: center["design"][name]
                    for name in ("compressor_blade_count", "combustor_injector_count", "turbine_blade_count")
                },
                "design_schema_hash": center["design_schema_hash"],
                "physics_bundle_hash": center["physics_bundle_hash"],
            })
    rows.sort(key=lambda row: row["center_design_id"])
    identity = {
        "config_sha256": _sha(root / CONFIG),
        "code_sha256": _sha(root / CODE),
        "l0_manifest_sha256": config["source_centers"]["manifest_sha256"],
        "l0_centers_sha256": config["source_centers"]["centers_sha256"],
        "l1_complete_manifest_sha256": config["source_pilot"]["complete_manifest_sha256"],
        "l1_labels_sha256": config["source_pilot"]["labels_sha256"],
        "design_schema_hash": centers[0]["design_schema_hash"],
        "physics_bundle_hash": centers[0]["physics_bundle_hash"],
    }
    if any(row["design_schema_hash"] != identity["design_schema_hash"]
           or row["physics_bundle_hash"] != identity["physics_bundle_hash"] for row in rows):
        raise V3LocalDataContractError("L0 center runtime identity changed")
    return rows, identity


def _manifest(rows: list[dict[str, Any]], identity: dict[str, Any], content: bytes) -> dict[str, Any]:
    return {
        "status": "future_center_split_frozen_before_future_labels",
        "identity": identity,
        "center_count": len(rows),
        "split_counts": dict(Counter(row["center_split"] for row in rows)),
        "stratum_split_counts": {
            str(stratum): dict(Counter(row["center_split"] for row in rows
                                      if row["l0_flow_root_count_numerical_only"] == stratum))
            for stratum in (1, 2)
        },
        "registry_sha256": hashlib.sha256(content).hexdigest(),
        "source_l0_l1_training_allowed": False,
        "future_labels_present": False,
        "future_label_qualification": "converged_finite_borrowed_domain_for_software_regression_only",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def verify_v3_local_data_contract(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    try:
        expected, identity = _expected(root)
        content = b"".join(_bytes(row) for row in expected)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if (
            _rows(output / "center_split_registry.jsonl") != expected
            or manifest != _manifest(expected, identity, content)
            or _sha(output / "center_split_registry.jsonl") != manifest["registry_sha256"]
        ):
            raise V3LocalDataContractError("V3 center split registry changed")
    except V3LocalDataContractError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalDataContractError("Cannot verify V3 local data contract") from exc
    return manifest


def freeze_v3_local_data_contract(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3LocalDataContractError(f"Refusing to overwrite V3 local data contract: {output}")
    rows, identity = _expected(root)
    content = b"".join(_bytes(row) for row in rows)
    manifest = _manifest(rows, identity, content)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_local_contract_", dir=output.parent))
    try:
        (temporary / "center_split_registry.jsonl").write_bytes(content)
        (temporary / "manifest.json").write_bytes(_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v3_local_data_contract(root, output)
