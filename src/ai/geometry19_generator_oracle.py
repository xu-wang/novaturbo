"""Direct-only oracle evaluation of Geometry19 conditional generators."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch
import yaml

from src.ai.generators import CFMGenerator, load_generator
from src.ai.geometry19_candidates import sample_device_invariant_sources
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.ai.geometry19_physics import evaluate_geometry19_candidate
from src.ai.geometry19_pipeline import build_target_mapping
from src.ai.geometry19_selection import ConstraintDominanceContract
from src.ai.geometry19_surrogate import sha256_file
from src.physics import ConstraintProfile, Environment, Geometry19, GeometryStateError, ModelConfig, derive_geometry_state


VERSION = "wp19_23_generator_core_oracle_v1"
OUTPUTS = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
_WORKER: dict[str, Any] = {}


class GeneratorOracleError(ValueError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _performance(row: Mapping[str, Any]) -> np.ndarray:
    perf = row["physics_outputs"]["performance"]
    return np.asarray([perf[name] for name in OUTPUTS], dtype=float)


def _condition_features(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[:, 1:] = np.log(result[:, 1:])
    return result


def _farthest_select(rows: Sequence[Mapping[str, Any]], count: int, mean: np.ndarray, std: np.ndarray) -> list[dict[str, Any]]:
    if count > len(rows):
        raise GeneratorOracleError("Target stratum is smaller than requested count")
    points = (_condition_features(np.asarray([_performance(row) for row in rows])) - mean) / std
    selected = [int(np.argmin(np.linalg.norm(points, axis=1)))]
    minimum = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < count:
        minimum[selected] = -1.0
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(points - points[index], axis=1))
    return [dict(rows[index]) for index in selected]


def _select_split(rows: Sequence[Mapping[str, Any]], train_points: np.ndarray, count: int, split: str) -> list[dict[str, Any]]:
    values = _condition_features(np.asarray([_performance(row) for row in rows]))
    mean = train_points.mean(axis=0)
    std = train_points.std(axis=0)
    std[std == 0.0] = 1.0
    normalized = (values - mean) / std
    tree = cKDTree((train_points - mean) / std)
    k = 2 if split == "train_diagnostic" else 1
    distances = np.asarray(tree.query(normalized, k=k)[0])
    if distances.ndim == 2:
        distances = distances[:, -1]
    order = np.argsort(distances, kind="stable")
    groups = np.array_split(order, 3)
    base, remainder = divmod(count, 3)
    chosen: list[dict[str, Any]] = []
    labels = ("dense", "mid", "sparse")
    for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
        take = base + int(index < remainder)
        pool = [rows[int(position)] for position in group]
        selected = _farthest_select(pool, take, mean, std)
        distance_by_sample = {
            int(rows[int(position)]["sample_id"]): float(distances[int(position)])
            for position in group
        }
        for row in selected:
            row["target_region"] = label
            row["train_neighbor_distance"] = distance_by_sample[int(row["sample_id"])]
            chosen.append(row)
    return chosen


def build_target_suite(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _read_jsonl(root / config["data"]["regression_dataset"])
    by_campaign: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_campaign.setdefault(str(row["campaign_id"]), []).append(row)
    train_rows = [row for campaign in config["data"]["train_campaigns"] for row in by_campaign[campaign]]
    train_points = _condition_features(np.asarray([_performance(row) for row in train_rows]))
    specifications = (
        ("train_diagnostic", train_rows, int(config["data"]["target_counts"]["train_diagnostic"])),
        ("development", by_campaign[config["data"]["validation_campaign"]], int(config["data"]["target_counts"]["development"])),
        ("locked_held_out", by_campaign[config["data"]["held_out_campaign"]], int(config["data"]["target_counts"]["locked_held_out"])),
    )
    suite: list[dict[str, Any]] = []
    for split, pool, count in specifications:
        for row in _select_split(pool, train_points, count, split):
            perf = row["physics_outputs"]["performance"]
            target = build_target_mapping(thrust_N=perf["thrust_N"], tsfc_kg_N_s=perf["tsfc_kg_N_s"], mass_kg=perf["mass_kg"], target_type="ID", expected_status="KNOWN_FEASIBLE")
            suite.append({
                "split": split,
                "target_region": row["target_region"],
                "source_campaign": row["campaign_id"],
                "sample_id": int(row["sample_id"]),
                "root_classification": row["solver_status"]["classification"],
                "engineering_feasible": bool(row["constraint_values"]["assessment"]["engineering_feasible"]),
                "train_neighbor_distance": row["train_neighbor_distance"],
                "target": target,
                "reference_x_19": row["x_19"],
            })
    historical = yaml.safe_load((root / config["data"]["historical_targets"]).read_text(encoding="utf-8"))
    test_by_id = {int(row["sample_id"]): row for row in by_campaign[config["data"]["held_out_campaign"]]}
    for spec in historical["targets"]:
        row = test_by_id[int(spec["sample_id"])]
        perf = row["physics_outputs"]["performance"]
        suite.append({
            "split": "historical_regression", "target_region": "historical", "source_campaign": row["campaign_id"],
            "sample_id": int(row["sample_id"]), "root_classification": row["solver_status"]["classification"],
            "engineering_feasible": bool(row["constraint_values"]["assessment"]["engineering_feasible"]),
            "train_neighbor_distance": None,
            "target": build_target_mapping(thrust_N=perf["thrust_N"], tsfc_kg_N_s=perf["tsfc_kg_N_s"], mass_kg=perf["mass_kg"], target_type="ID", expected_status="KNOWN_FEASIBLE"),
            "reference_x_19": row["x_19"],
        })
    return suite


def _worker_init(root: str) -> None:
    repo = Path(root)
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    _WORKER.update({
        "acceptance": contract.diagnostic_acceptance,
        "model_config": ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"]),
        "constraint_profile": ConstraintProfile.from_yaml(repo / contract.payload["contracts"]["constraint_profile"]),
        "dominance": ConstraintDominanceContract.from_yaml(repo / contract.payload["contracts"]["constraint_dominance"]),
        "environment": Environment(),
    })


def _physics_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    target = Geometry19InverseTarget.from_mapping(task["target"])
    result = evaluate_geometry19_candidate(
        int(task["candidate_id"]), str(task["run_id"]), task["geometry19"], str(task["design_hash"]), target,
        _WORKER["acceptance"], _WORKER["model_config"], _WORKER["constraint_profile"], _WORKER["environment"],
        constraint_dominance_contract=_WORKER["dominance"],
    )
    return {
        "run_id": task["run_id"], "target_id": target.target_id, "candidate_id": task["candidate_id"],
        "design_hash": task["design_hash"], "physics_acceptable": result["physics_acceptable"],
        "solver_converged": result["solver_converged"], "root_classification": result["root_classification"],
        "solver_failure_code": result["solver_failure_code"], "performance": result["performance"],
        "relative_errors": result["relative_errors"], "hit_alignment": result["hit_alignment"],
        "development_diagnostic_hit": result["development_diagnostic_hit"],
    }


def _filter_samples(
    normalized: np.ndarray,
    arm: str,
    target_id: str,
    source_seed: int,
    schema: Mapping[str, Any],
    model_config: ModelConfig,
    discrete_tolerance_count: float | None = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = list(schema["design_variables"])
    names = [item["name"] for item in fields]
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=float)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=float)
    integer = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    run_id = f"{arm}:{target_id}:{source_seed}"
    candidates: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate_id, vector in enumerate(normalized):
        failure = None
        geometry = None
        design_hash = None
        if not np.isfinite(vector).all():
            failure = "NON_FINITE_GENERATION"
        elif np.any(vector < -1.0) or np.any(vector > 1.0):
            failure = "OUT_OF_SCHEMA_BOUNDS"
        else:
            physical = lower + 0.5 * (vector + 1.0) * (upper - lower)
            rounded = np.rint(physical[integer])
            if discrete_tolerance_count is not None and np.any(
                np.abs(physical[integer] - rounded)
                > float(discrete_tolerance_count) + 1e-12
            ):
                failure = "INVALID_DISCRETE_COORDINATE"
            else:
                physical[integer] = rounded
                mapping = {name: int(physical[index]) if index in integer else float(physical[index]) for index, name in enumerate(names)}
                try:
                    design = Geometry19.from_mapping(mapping)
                    derive_geometry_state(design, model_config)
                    geometry = design.as_mapping()
                    design_hash = sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if design_hash in seen:
                        failure = "DUPLICATE_CANDIDATE"
                    else:
                        seen.add(design_hash)
                except (TypeError, ValueError, GeometryStateError):
                    failure = "GEOMETRY_MAPPING_FAILED"
        row = {"run_id": run_id, "arm_id": arm, "target_id": target_id, "source_seed": source_seed, "candidate_id": candidate_id, "failure_code": failure, "design_hash": design_hash, "geometry19": geometry}
        candidates.append(row)
        if failure is None:
            tasks.append({**row})
    return candidates, tasks


def _prefix_metrics(rows: Sequence[Mapping[str, Any]], prefixes: Sequence[int]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for prefix in prefixes:
        subset = rows[:prefix]
        rhos = [float(row["rho"]) for row in subset if row.get("rho") is not None]
        values[str(prefix)] = {
            "oracle_hit": any(value <= 1.0 for value in rhos), "best_true_rho": min(rhos) if rhos else None,
            "strict_hits": sum(value <= 1.0 for value in rhos), "near_hits_2": sum(value <= 2.0 for value in rhos),
            "near_hits_5": sum(value <= 5.0 for value in rhos), "hit_density": sum(value <= 1.0 for value in rhos) / prefix,
            "physics_acceptable": len(rhos),
        }
    return values


def _aggregate(runs: Sequence[Mapping[str, Any]], split: str, arm: str, prefixes: Sequence[int]) -> dict[str, Any]:
    selected = [run for run in runs if run["split"] == split and run["arm_id"] == arm]
    result: dict[str, Any] = {"runs": len(selected), "targets": len({run["target_id"] for run in selected})}
    for prefix in prefixes:
        metrics = [run["prefix_metrics"][str(prefix)] for run in selected]
        best = [row["best_true_rho"] for row in metrics if row["best_true_rho"] is not None]
        result[str(prefix)] = {
            "oracle_hit_rate": sum(row["oracle_hit"] for row in metrics) / len(metrics),
            "median_best_true_rho": float(np.median(best)) if best else None,
            "p90_best_true_rho": float(np.percentile(best, 90)) if best else None,
            "median_hit_density": float(np.median([row["hit_density"] for row in metrics])),
            "mean_physics_acceptable": float(np.mean([row["physics_acceptable"] for row in metrics])),
        }
    stability_key = "256" if 256 in prefixes else str(max(prefixes))
    target_hits: dict[str, int] = {}
    for run in selected:
        target_hits[run["target_id"]] = target_hits.get(run["target_id"], 0) + int(run["prefix_metrics"][stability_key]["oracle_hit"])
    result["stable_target_fraction_256"] = sum(value >= 2 for value in target_hits.values()) / len(target_hits) if target_hits else None
    return result


def _render_report(summary: Mapping[str, Any]) -> str:
    def number(value: Any, spec: str = ".3f") -> str:
        return "N/A" if value is None else format(float(value), spec)

    lines = ["# Geometry19 生成器本体 Oracle 实验报告", "", f"> 生成时间：{summary['generated_at_utc']}", "> 范围：direct-only；未使用 D-Flow、reranker、gradient refinement 或 NSGA-II。", "", "## 主结果", "", "| Arm | Split | Hit@256 | Hit@512 | median best rho@256 | P90 best rho@256 | median hit density@256 | stable targets |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, splits in summary["aggregate"].items():
        for split in ("train_diagnostic", "development", "locked_held_out", "historical_regression"):
            row = splits[split]
            keys = sorted((int(key) for key in row if str(key).isdigit()))
            key256 = "256" if 256 in keys else str(keys[-1])
            key512 = "512" if 512 in keys else str(keys[-1])
            p256, p512 = row[key256], row[key512]
            lines.append(f"| {arm} | {split} | {p256['oracle_hit_rate']:.1%} | {p512['oracle_hit_rate']:.1%} | {number(p256['median_best_true_rho'])} | {number(p256['p90_best_true_rho'])} | {p256['median_hit_density']:.2%} | {number(row['stable_target_fraction_256'], '.1%')} |")
    lines += ["", "## 严格候选漏斗（全 512 池）", "", "| Arm | raw | bounds fail | discrete fail | geometry fail | valid | physics acceptable | strict hits |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, row in summary["funnel"].items():
        lines.append(f"| {arm} | {row['raw']} | {row.get('OUT_OF_SCHEMA_BOUNDS',0)} | {row.get('INVALID_DISCRETE_COORDINATE',0)} | {row.get('GEOMETRY_MAPPING_FAILED',0)} | {row['prefilter_valid']} | {row['physics_acceptable']} | {row['strict_hits']} |")
    lines += ["", "## Development Gate", ""]
    for key, value in summary["gate"].items():
        lines.append(f"- {key}: **{value}**")
    lines += ["", "本报告只评价已有生成器。没有开展结构、loss、采样框架或后处理改进实验。", ""]
    return "\n".join(lines)


def run_generator_oracle(root: str | Path, config_path: str | Path, *, device: str | None = None, output_root: str | Path | None = None, smoke: bool = False) -> dict[str, Any]:
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config["protocol_version"] != VERSION:
        raise GeneratorOracleError("Protocol version mismatch")
    runtime = Path(output_root).resolve() if output_root else repo / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise GeneratorOracleError(f"Refusing to overwrite {runtime}")
    runtime.mkdir(parents=True)
    suite = build_target_suite(repo, config)
    if smoke:
        keep = []
        for split in ("train_diagnostic", "development", "locked_held_out", "historical_regression"):
            keep.extend([row for row in suite if row["split"] == split][:1])
        suite = keep
    _write_jsonl(runtime / config["artifacts"]["target_suite"], suite)
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    schema = yaml.safe_load((repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8"))
    model_config = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    prefixes = [16, 32] if smoke else list(config["sampling"]["nested_prefixes"])
    maximum = max(prefixes)
    seeds = [int(config["sampling"]["source_seeds"][0])] if smoke else [int(value) for value in config["sampling"]["source_seeds"]]
    arms = list(config["models"][:1]) if smoke else list(config["models"])
    candidate_rows: list[dict[str, Any]] = []
    physics_tasks: list[dict[str, Any]] = []
    generated_seconds: dict[str, float] = {}
    runtime_device = device or config["execution"]["device"]
    for arm in arms:
        checkpoint = repo / arm["checkpoint"]
        generator = load_generator(checkpoint, device=runtime_device)
        if not isinstance(generator, CFMGenerator):
            raise GeneratorOracleError("Checkpoint is not CFM")
        started = perf_counter()
        for target_row in suite:
            target = Geometry19InverseTarget.from_mapping(target_row["target"])
            condition = np.asarray([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg], dtype=float)
            normalized_condition = torch.as_tensor(generator.normalizer.transform_conditions(condition), dtype=generator.dtype, device=generator.device)
            for seed in seeds:
                sources = sample_device_invariant_sources(generator, maximum, seed)
                with torch.no_grad():
                    samples = integrate_cfm_sources(generator, sources, normalized_condition).cpu().numpy()
                candidates, tasks = _filter_samples(samples, arm["arm_id"], target.target_id, seed, schema, model_config)
                for row in candidates:
                    row["split"] = target_row["split"]
                    row["target_region"] = target_row["target_region"]
                for task in tasks:
                    task["target"] = target.as_mapping()
                candidate_rows.extend(candidates)
                physics_tasks.extend(tasks)
        generated_seconds[arm["arm_id"]] = perf_counter() - started
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    workers = 2 if smoke else int(config["execution"]["physics_workers"])
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(str(repo),)) as executor:
        physics_rows = list(executor.map(_physics_worker, physics_tasks, chunksize=int(config["execution"]["physics_chunksize"])))
    lookup = {(row["run_id"], int(row["candidate_id"])): row for row in physics_rows}
    targets_by_id = {row["target"]["target_id"]: row for row in suite}
    runs: list[dict[str, Any]] = []
    for arm in arms:
        for target_id, target_row in targets_by_id.items():
            for seed in seeds:
                run_id = f"{arm['arm_id']}:{target_id}:{seed}"
                rows = [row for row in candidate_rows if row["run_id"] == run_id]
                for row in rows:
                    physics = lookup.get((run_id, int(row["candidate_id"])))
                    row["physics_acceptable"] = bool(physics and physics["physics_acceptable"])
                    row["rho"] = (physics.get("hit_alignment") or {}).get("rho") if physics else None
                runs.append({"run_id": run_id, "arm_id": arm["arm_id"], "target_id": target_id, "split": target_row["split"], "target_region": target_row["target_region"], "source_seed": seed, "prefix_metrics": _prefix_metrics(rows, prefixes)})
    funnel: dict[str, dict[str, int]] = {}
    for arm in arms:
        rows = [row for row in candidate_rows if row["arm_id"] == arm["arm_id"]]
        counts: dict[str, int] = {"raw": len(rows), "prefilter_valid": sum(row["failure_code"] is None for row in rows), "physics_acceptable": sum(row.get("physics_acceptable", False) for row in rows), "strict_hits": sum(row.get("rho") is not None and row["rho"] <= 1.0 for row in rows)}
        for row in rows:
            if row["failure_code"]:
                counts[row["failure_code"]] = counts.get(row["failure_code"], 0) + 1
        funnel[arm["arm_id"]] = counts
    aggregate = {arm["arm_id"]: {split: _aggregate(runs, split, arm["arm_id"], prefixes) for split in ("train_diagnostic", "development", "locked_held_out", "historical_regression")} for arm in arms}
    thresholds = config["development_thresholds"]
    best_arm = arms[-1]["arm_id"]
    held = aggregate[best_arm]["locked_held_out"]
    gate_checks = {
        "oracle_hit_256": held.get("256", {}).get("oracle_hit_rate", 0) >= float(thresholds["oracle_hit_256"]),
        "oracle_hit_512": held.get("512", {}).get("oracle_hit_rate", 0) >= float(thresholds["oracle_hit_512"]),
        "median_best_rho_256": (held.get("256", {}).get("median_best_true_rho") or math.inf) <= float(thresholds["median_best_true_rho_256"]),
        "p90_best_rho_256": (held.get("256", {}).get("p90_best_true_rho") or math.inf) <= float(thresholds["p90_best_true_rho_256"]),
        "stable_targets_256": (held.get("stable_target_fraction_256") or 0) >= float(thresholds["stable_target_fraction"]),
        "hit_density_256": held.get("256", {}).get("median_hit_density", 0) >= float(thresholds["median_strict_hit_density_256"]),
    }
    summary = {
        "protocol_version": VERSION, "status": "SMOKE_PASS" if smoke else "COMPLETE", "formal_use_allowed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "direct_only_generator_analysis",
        "target_count": len(suite), "source_seeds": seeds, "prefixes": prefixes, "generated_seconds": generated_seconds,
        "physics_task_count": len(physics_tasks), "aggregate": aggregate, "funnel": funnel,
        "gate_checks": gate_checks, "gate": {"overall": "PASS" if all(gate_checks.values()) else "FAIL", **{key: "PASS" if value else "FAIL" for key, value in gate_checks.items()}},
        "hashes": {"config": sha256_file(source), "schema": sha256_file(repo / contract.payload["contracts"]["design_schema"]), "calibration": model_config.calibration_pack_hash, "checkpoints": {arm["arm_id"]: sha256_file(repo / arm["checkpoint"]) for arm in arms}, "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()},
        "excluded": ["dflow", "reranker", "gradient_refinement", "nsga2", "projection", "refill", "new_training"],
    }
    _write_jsonl(runtime / config["artifacts"]["candidate_ledger"], candidate_rows)
    _write_jsonl(runtime / config["artifacts"]["physics_ledger"], physics_rows)
    _write_json(runtime / config["artifacts"]["summary"], summary)
    (runtime / config["artifacts"]["report"]).write_text(_render_report(summary), encoding="utf-8")
    return summary
