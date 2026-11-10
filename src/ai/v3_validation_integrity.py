"""Dependency locking and complete ledger verification for V3 software checks."""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_hashes(root: Path) -> dict[str, str]:
    """Lock the transitive repository Python imports, including package initializers."""
    pending = ["src.ai.v3_fresh_validation_l6h", "src.ai.v3_validation_integrity"]
    found = set()
    while pending:
        module = pending.pop()
        candidates = [root / (module.replace(".", "/") + ".py"),
                      root / module.replace(".", "/") / "__init__.py"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None or path in found:
            continue
        found.add(path)
        parts = module.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names if a.name.startswith("src."))
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    package = parts if path.name == "__init__.py" else parts[:-1]
                    name = ".".join(package[:len(package) - node.level + 1] + ([name] if name else []))
                if name == "src" or name.startswith("src."):
                    pending.append(name)
                    pending.extend(name + "." + a.name for a in node.names)
    return {str(p.relative_to(root)): sha(p) for p in sorted(found)}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def verify_complete(output: Path, selection: dict) -> dict:
    """Check final hashes, row identities, F choices, and every reported arm count."""
    complete = json.loads((output / "complete_manifest.json").read_text())
    for key, name in (("selection_manifest_sha256", "selection_manifest.json"),
                      ("physics_ledger_sha256", "physics_ledger.jsonl"),
                      ("f_recompute_sha256", "f_recompute.jsonl")):
        if complete[key] != sha(output / name):
            raise ValueError("Final artifact hash changed: " + name)
    physics = read_rows(output / "physics_ledger.jsonl")
    selected = read_rows(output / "selection_ledger.jsonl")
    recompute = read_rows(output / "f_recompute.jsonl")
    identity = lambda t: (t["target_id"], t["arm"], t["design_id"], t["selection_rank"], t["route_center_id"])
    if ([identity(t) for t in physics] != [identity(t) for t in selected]
            or complete["targets"] != selection["targets"]
            or complete["routable_targets"] != selection["routable_targets"]
            or complete["raw_slots"] != selection["raw_slots"]
            or complete["physics_evaluations"] != len(physics)
            or complete["f_recomputations"] != len(recompute)
            or complete["formal_use_allowed"] is not False
            or complete["deployment_allowed"] is not False):
        raise ValueError("Final population or identity changed")
    grouped = defaultdict(list)
    target_path = output / "target_labels.jsonl"
    targets = {t["target_id"]: t for t in read_rows(target_path)} if target_path.exists() else {}
    for p, s in zip(physics, selected):
        if any(p[k] != s[k] for k in ("physics_bundle_hash", "design_schema_hash")):
            raise ValueError("Final physics identity changed")
        if p["software_hit_5pct"] != (p["actual_rho_5pct"] is not None and p["actual_rho_5pct"] <= 1):
            raise ValueError("Final hit flag changed")
        if targets and p["performance"] is not None:
            values = p["performance"]
            expected_rho = max(abs(values[n] / targets[p["target_id"]]["target_performance"][n] - 1) / .05 for n in ("thrust_N", "tsfc_kg_N_s", "mass_kg"))
            if not math.isfinite(expected_rho) or not math.isclose(expected_rho, p["actual_rho_5pct"], abs_tol=1e-12):
                raise ValueError("Final performance/rho changed")
        grouped[(p["target_id"], p["arm"])].append(p)
    expected_f = []
    for (target, arm), group in sorted(grouped.items()):
        valid = [p for p in group if p["actual_rho_5pct"] is not None]
        if not valid:
            continue
        best = min(valid, key=lambda p: (p["actual_rho_5pct"], p["design_id"]))
        expected_f.append((target, arm, best["design_id"], best["actual_rho_5pct"]))
    if [(f["target_id"], f["arm"], f["design_id"], f["original_rho_5pct"]) for f in recompute] != expected_f:
        raise ValueError("Final F best-candidate population changed")
    if any(not f["independent_v3_recompute_equal"] or f["original_rho_5pct"] != f["recomputed_rho_5pct"] for f in recompute):
        raise ValueError("Final F recompute differs")
    originals = {(p["target_id"], p["arm"], p["design_id"]): p for p in physics}
    for f in recompute:
        if targets and f.get("recomputed_result") != originals[(f["target_id"], f["arm"], f["design_id"])]:
            raise ValueError("Final F complete result differs")
    if set(complete["arms"]) != {"gated_top1", "gated_top2", "nearest_center_without_gate"}:
        raise ValueError("Final arm population changed")
    for arm, summary in complete["arms"].items():
        group = [p for p in physics if p["arm"] == arm]
        hits = sorted({p["target_id"] for p in group if p["software_hit_5pct"]})
        expected = {"selected_candidates": len(group), "targets_evaluated": len({p["target_id"] for p in group}),
                    "solver_converged": sum(p["solver_converged"] for p in group),
                    "diagnostic_domain_valid": sum(p["diagnostic_domain_valid"] for p in group),
                    "engine_feasible": sum(p["engine_feasible"] for p in group),
                    "software_hit_targets_5pct": len(hits), "software_hit_target_ids_5pct": hits,
                    "f_recompute_count": sum(f["arm"] == arm for f in recompute),
                    "f_recompute_equal_count": sum(f["arm"] == arm and f["independent_v3_recompute_equal"] for f in recompute)}
        if summary != expected:
            raise ValueError("Final arm summary changed")
    return complete
