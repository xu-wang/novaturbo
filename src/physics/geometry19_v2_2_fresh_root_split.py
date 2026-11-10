"""Freeze a globally root-disjoint split with fresh low-mass confirmation roots."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.physics.geometry19_v2_2_dataset import (
    Geometry19V22DatasetError,
    _artifact_path,
    _hash_file,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)


FRESH_SPLIT_VERSION = "geometry19_v2_2_fresh_low_mass_confirmation_split_v1"
ROLE_NAMES = ("train", "validation", "confirmation")


def _load_protocol(root: Path, source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(source)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("dataset_version") != FRESH_SPLIT_VERSION:
        raise Geometry19V22DatasetError("Fresh-root split protocol version mismatch")
    if protocol.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Fresh-root split cannot be formal")
    if protocol.get("grouping_key") != "root_seed_id":
        raise Geometry19V22DatasetError("Fresh split must use global root_seed_id grouping")
    return path, protocol


def _role_roots(protocol: Mapping[str, Any]) -> dict[str, set[str]]:
    validation = {str(value) for value in protocol["roles"]["validation_roots"]}
    confirmation = {
        str(value)
        for value in protocol["roles"]["confirmation_roots_by_segment"].values()
    }
    if not validation or not confirmation or validation & confirmation:
        raise Geometry19V22DatasetError("Validation/confirmation root roles overlap or are empty")
    return {"validation": validation, "confirmation": confirmation}


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    segments = sorted({str(row["assigned_segment"]) for row in rows})
    return {
        "rows": len(rows),
        "global_roots": len({str(row["root_seed_id"]) for row in rows}),
        "by_segment": {
            segment: {
                "rows": sum(row["assigned_segment"] == segment for row in rows),
                "roots": len(
                    {
                        str(row["root_seed_id"])
                        for row in rows
                        if row["assigned_segment"] == segment
                    }
                ),
            }
            for segment in segments
        },
    }


def freeze_fresh_root_split(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_source, protocol = _load_protocol(root, protocol_path)
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise Geometry19V22DatasetError(f"Refusing to overwrite split: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_spec = protocol["source_dataset"]
    source_path = (root / source_spec["path"]).resolve()
    if _hash_file(source_path) != source_spec["sha256"]:
        raise Geometry19V22DatasetError("Combined source hash mismatch")
    prior_spec = protocol["prior_exposed_dataset"]
    prior_path = (root / prior_spec["path"]).resolve()
    if _hash_file(prior_path) != prior_spec["sha256"]:
        raise Geometry19V22DatasetError("Prior exposed dataset hash mismatch")
    rows = _read_jsonl(source_path)
    prior_rows = _read_jsonl(prior_path)
    if len({row["design_hash"] for row in rows}) != len(rows):
        raise Geometry19V22DatasetError("Combined source contains duplicate designs")

    fixed = _role_roots(protocol)
    all_roots = {str(row["root_seed_id"]) for row in rows}
    if not (fixed["validation"] | fixed["confirmation"]) <= all_roots:
        raise Geometry19V22DatasetError("A frozen held-out root is absent from the source")
    prior_roots = {str(row["root_seed_id"]) for row in prior_rows}
    if fixed["confirmation"] & prior_roots:
        raise Geometry19V22DatasetError("Confirmation roots are not globally fresh")

    role_by_root = {
        root_id: (
            "confirmation"
            if root_id in fixed["confirmation"]
            else "validation"
            if root_id in fixed["validation"]
            else "train"
        )
        for root_id in all_roots
    }
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ROLE_NAMES}
    for row in rows:
        split_rows[role_by_root[str(row["root_seed_id"])]].append(row)
    for role in ROLE_NAMES:
        split_rows[role].sort(
            key=lambda row: (row["assigned_segment"], row["root_seed_id"], row["design_hash"])
        )

    observed_roles: dict[str, set[str]] = defaultdict(set)
    for role, role_rows in split_rows.items():
        for row in role_rows:
            observed_roles[str(row["root_seed_id"])].add(role)
    if any(len(roles) != 1 for roles in observed_roles.values()):
        raise Geometry19V22DatasetError("Global root leakage across split roles")

    target_segments = tuple(protocol["target_segments"])
    gate = protocol["split_gate"]
    summaries = {role: _summarize(role_rows) for role, role_rows in split_rows.items()}
    gate_pass = summaries["train"]["rows"] >= int(gate["minimum_train_rows"])
    for role in ("validation", "confirmation"):
        for segment in target_segments:
            values = summaries[role]["by_segment"].get(segment, {"rows": 0, "roots": 0})
            gate_pass = gate_pass and values["rows"] >= int(
                gate[f"minimum_{role}_rows_per_target_segment"]
            )
            gate_pass = gate_pass and values["roots"] >= int(
                gate[f"minimum_{role}_roots_per_target_segment"]
            )

    split_paths: dict[str, Path] = {}
    for role, role_rows in split_rows.items():
        path = output / "splits" / f"{role}.jsonl"
        _write_jsonl(path, role_rows)
        split_paths[role] = path
    manifest = {
        "dataset_version": FRESH_SPLIT_VERSION,
        "status": "FRESH_ROOT_SPLIT_PASS" if gate_pass else "FRESH_ROOT_SPLIT_FAIL",
        "formal_use_allowed": False,
        "training_ready": gate_pass,
        "scope": "low_mass_corrective_surrogate_experiment_only",
        "full_domain_confirmation_ready": False,
        "grouping_key": "root_seed_id",
        "prior_exposed_rows_reclassified_as_development": len(prior_rows),
        "target_segments": list(target_segments),
        "roles": {
            "validation_roots": sorted(fixed["validation"]),
            "confirmation_roots": sorted(fixed["confirmation"]),
            "root_assignments": role_by_root,
        },
        "counts": summaries,
        "split_gate": gate,
        "hashes": {
            "protocol_hash": _hash_file(protocol_source),
            "source_dataset_hash": _hash_file(source_path),
            "prior_exposed_dataset_hash": _hash_file(prior_path),
            **{
                f"{role}_split_hash": _hash_file(path)
                for role, path in split_paths.items()
            },
        },
        "paths": {
            "source_dataset": _artifact_path(source_path, root),
            "splits": {
                role: _artifact_path(path, root) for role, path in split_paths.items()
            },
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
