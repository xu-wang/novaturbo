"""Verification helpers for the frozen Stage-II active-10D baseline."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_git_ancestor(repo_root: Path, source_commit: str) -> Dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "head": head,
        "source_commit": source_commit,
        "source_is_ancestor": completed.returncode == 0,
    }


def _csv_summary(path: Path, check: Mapping[str, Any]) -> Dict[str, Any]:
    row_count = 0
    requested_columns = tuple(check.get("value_counts", {}).keys())
    counts = {column: Counter() for column in requested_columns}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        for row in reader:
            row_count += 1
            for column in requested_columns:
                counts[column][str(row[column])] += 1
    return {
        "row_count": row_count,
        "columns": columns,
        "value_counts": {
            column: dict(column_counts) for column, column_counts in counts.items()
        },
    }


def verify_manifest(
    manifest_path: Path,
    *,
    repo_root: Optional[Path] = None,
    check_git: bool = True,
) -> Dict[str, Any]:
    """Verify frozen files and semantic CSV contracts from a YAML manifest."""

    manifest_path = Path(manifest_path).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Baseline manifest must contain a YAML mapping")
    if manifest.get("immutable") is not True:
        raise ValueError("Baseline manifest must declare immutable: true")

    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    issues = []
    verified = []

    for artifact in manifest.get("artifacts", []):
        relative = Path(str(artifact["path"]))
        path = root / relative
        if not path.is_file():
            issues.append({"path": str(relative), "error": "missing"})
            continue
        actual_size = path.stat().st_size
        expected_size = int(artifact["bytes"])
        if actual_size != expected_size:
            issues.append(
                {
                    "path": str(relative),
                    "error": "size_mismatch",
                    "expected": expected_size,
                    "actual": actual_size,
                }
            )
            continue
        actual_sha256 = sha256_file(path)
        expected_sha256 = str(artifact["sha256"])
        if actual_sha256 != expected_sha256:
            issues.append(
                {
                    "path": str(relative),
                    "error": "sha256_mismatch",
                    "expected": expected_sha256,
                    "actual": actual_sha256,
                }
            )
            continue
        verified.append(str(relative))

    semantic_results: Dict[str, Any] = {}
    for check in manifest.get("semantic_checks", []):
        if check.get("kind") != "csv":
            issues.append(
                {
                    "path": str(check.get("path", "")),
                    "error": "unsupported_semantic_check",
                    "kind": check.get("kind"),
                }
            )
            continue
        relative = Path(str(check["path"]))
        path = root / relative
        if not path.is_file():
            issues.append({"path": str(relative), "error": "semantic_file_missing"})
            continue
        summary = _csv_summary(path, check)
        semantic_results[str(relative)] = summary
        expected_rows = check.get("row_count")
        if expected_rows is not None and summary["row_count"] != int(expected_rows):
            issues.append(
                {
                    "path": str(relative),
                    "error": "row_count_mismatch",
                    "expected": int(expected_rows),
                    "actual": summary["row_count"],
                }
            )
        for column, expected in check.get("value_counts", {}).items():
            actual = summary["value_counts"].get(column, {})
            normalized_expected = {str(key): int(value) for key, value in expected.items()}
            if actual != normalized_expected:
                issues.append(
                    {
                        "path": str(relative),
                        "error": "value_counts_mismatch",
                        "column": column,
                        "expected": normalized_expected,
                        "actual": actual,
                    }
                )

    git_result: Dict[str, Any] = {"checked": False}
    if check_git:
        git_result = {
            "checked": True,
            **_verify_git_ancestor(root, str(manifest["code"]["source_commit"])),
        }
        if not git_result["source_is_ancestor"]:
            issues.append(
                {
                    "error": "source_commit_not_ancestor",
                    "source_commit": git_result["source_commit"],
                    "head": git_result["head"],
                }
            )

    return {
        "ok": not issues,
        "manifest": str(manifest_path),
        "baseline_id": manifest.get("baseline_id"),
        "checked_artifacts": len(verified),
        "verified_artifacts": verified,
        "semantic_checks": semantic_results,
        "git": git_result,
        "issues": issues,
    }

