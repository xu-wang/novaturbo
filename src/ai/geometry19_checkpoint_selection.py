"""Downstream-aware checkpoint selection for the Geometry19 C generator."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.ai import geometry19_mixed_cfm as _mixed_registration  # noqa: F401
from src.ai.generators import load_generator
from src.ai.geometry19_generator_oracle import (
    _condition_features,
    _filter_samples,
    _performance,
    _physics_worker,
    _read_jsonl,
    _select_split,
    _worker_init,
    _write_json,
    _write_jsonl,
)
from src.ai.geometry19_generator_failure import _map_candidates
from src.ai.geometry19_candidates import sample_device_invariant_sources
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.ai.geometry19_pipeline import build_target_mapping
from src.ai.geometry19_surrogate import sha256_file
from src.physics import ModelConfig


VERSION = "wp19_23_c_checkpoint_selection_v1"


def freeze_confirmation_targets(repo: Path, config: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    if output.exists():
        return _read_jsonl(output)
    source_suite = _read_jsonl(repo / config["evaluation"]["target_suite"])
    used = {int(row["sample_id"]) for row in source_suite}
    records = _read_jsonl(repo / config["data"]["regression_dataset"])
    by_campaign: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_campaign.setdefault(str(row["campaign_id"]), []).append(row)
    train_rows = [
        row for campaign in config["data"]["campaign_split"]["train"]
        for row in by_campaign[campaign]
    ]
    train_points = _condition_features(np.asarray([_performance(row) for row in train_rows]))
    test_campaign = str(config["data"]["campaign_split"]["test"][0])
    available = [row for row in by_campaign[test_campaign] if int(row["sample_id"]) not in used]
    selected = _select_split(
        available, train_points,
        int(config["checkpoint_selection"]["confirmation_target_count"]),
        "locked_held_out",
    )
    suite = []
    for row in selected:
        performance = row["physics_outputs"]["performance"]
        suite.append({
            "split": "new_locked_confirmation",
            "target_region": row["target_region"],
            "source_campaign": row["campaign_id"],
            "sample_id": int(row["sample_id"]),
            "train_neighbor_distance": float(row["train_neighbor_distance"]),
            "target": build_target_mapping(
                thrust_N=performance["thrust_N"],
                tsfc_kg_N_s=performance["tsfc_kg_N_s"],
                mass_kg=performance["mass_kg"],
                target_type="ID", expected_status="KNOWN_FEASIBLE",
            ),
            "reference_x_19": row["x_19"],
        })
    _write_jsonl(output, suite)
    return suite


def _evaluate_checkpoint(
    repo: Path, checkpoint: Path, targets: Sequence[Mapping[str, Any]],
    seeds: Sequence[int], count: int, device: str, workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    schema = yaml.safe_load((repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8"))
    physics_model = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    generator = load_generator(checkpoint, device=device)
    runs = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(str(repo),)) as executor:
        for target_row in targets:
            target = Geometry19InverseTarget.from_mapping(target_row["target"])
            physical_condition = np.asarray([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg])
            normalized_condition = torch.as_tensor(
                generator.normalizer.transform_conditions(physical_condition),
                dtype=generator.dtype, device=generator.device,
            )
            for seed in seeds:
                samples = generator.sample(normalized_condition, count, seed=seed, space="normalized").values.numpy()
                candidates, tasks = _filter_samples(samples, "c_dequantized", target.target_id, seed, schema, physics_model)
                for task in tasks:
                    task["target"] = target.as_mapping()
                physics = list(executor.map(_physics_worker, tasks, chunksize=32))
                by_id = {int(row["candidate_id"]): row for row in physics}
                for row in candidates:
                    result = by_id.get(int(row["candidate_id"]))
                    row["rho"] = (result.get("hit_alignment") or {}).get("rho") if result else None
                rhos = [float(row["rho"]) for row in candidates if row.get("rho") is not None]
                runs.append({
                    "target_id": target.target_id, "target_region": target_row["target_region"],
                    "source_seed": int(seed), "oracle_hit": bool(rhos and min(rhos) <= 1.0),
                    "best_true_rho": min(rhos) if rhos else None,
                    "physics_acceptable": len(rhos),
                    "strict_hits": sum(value <= 1.0 for value in rhos),
                })
    finite = [row["best_true_rho"] for row in runs if row["best_true_rho"] is not None]
    aggregate = {
        "runs": len(runs),
        "hit_rate": float(np.mean([row["oracle_hit"] for row in runs])),
        "median_best_rho": float(np.median(finite)),
        "p90_best_rho": float(np.percentile(finite, 90)),
        "mean_physics_acceptable": float(np.mean([row["physics_acceptable"] for row in runs])),
        "strict_hits": int(sum(row["strict_hits"] for row in runs)),
    }
    return aggregate, runs


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    # max() implements: hit first, then lower P90, median, and validation loss.
    return (
        float(row["hit_rate"]),
        -float(row["p90_best_rho"]),
        -float(row["median_best_rho"]),
        -float(row["validation_loss"]),
    )


def _render(summary: Mapping[str, Any]) -> str:
    selected = summary["selected"]
    lines = [
        "# C Dequantized-CFM下游指标Checkpoint选择",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> checkpoint选择仅使用development目标；新确认目标未参与本报告中的选择。",
        "",
        "## 选择结果",
        "",
        f"- selected epoch: **{selected['epoch']}**",
        f"- development Hit@128: **{selected['hit_rate']:.1%}**",
        f"- development median best rho: **{selected['median_best_rho']:.3f}**",
        f"- development P90 best rho: **{selected['p90_best_rho']:.3f}**",
        f"- validation loss: {selected['validation_loss']:.6f}",
        "",
        "## 全快照结果",
        "",
        "| Epoch | val loss | Hit@128 | median rho | P90 rho | acceptable |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["snapshots"]:
        marker = " **selected**" if row["epoch"] == selected["epoch"] else ""
        lines.append(
            f"| {row['epoch']}{marker} | {row['validation_loss']:.6f} | {row['hit_rate']:.1%} | "
            f"{row['median_best_rho']:.3f} | {row['p90_best_rho']:.3f} | {row['mean_physics_acceptable']:.1f} |"
        )
    corr = summary["correlations"]
    lines += [
        "", "## 指标相关性", "",
        f"- val loss vs Hit@128 Spearman: {corr['val_loss_vs_hit']:.3f}",
        f"- val loss vs median rho Spearman: {corr['val_loss_vs_median_rho']:.3f}",
        f"- val loss vs P90 rho Spearman: {corr['val_loss_vs_p90_rho']:.3f}",
        "", "本报告未使用旧held-out或新确认目标选择checkpoint。", "",
    ]
    return "\n".join(lines)


def run_checkpoint_selection(root: str | Path, config_path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    from scipy.stats import spearmanr

    repo = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).resolve().read_text(encoding="utf-8"))
    section = config["checkpoint_selection"]
    if section["protocol_version"] != VERSION:
        raise ValueError("Checkpoint selection protocol mismatch")
    runtime = repo / section["runtime_root"]
    if runtime.exists():
        raise ValueError(f"Refusing to overwrite checkpoint selection runtime: {runtime}")
    runtime.mkdir(parents=True)
    confirmation_path = runtime / "new_locked_confirmation_targets.jsonl"
    confirmation = freeze_confirmation_targets(repo, config, confirmation_path)
    existing = _read_jsonl(repo / config["evaluation"]["target_suite"])
    targets = [row for row in existing if row["split"] == "development"]
    if len(targets) != int(section["development_target_count"]):
        raise ValueError("Development target count mismatch")
    arm = next(row for row in config["arms"] if row["arm_id"] == "c_dequantized")
    weights = repo / arm["artifacts"]["weights_root"] / "checkpoint_campaign"
    snapshots = sorted((weights / "snapshots").glob("epoch_*.pt"))
    expected = list(range(int(section["snapshot_every"]), 401, int(section["snapshot_every"])))
    if [int(path.stem.split("_")[-1]) for path in snapshots] != expected:
        raise ValueError("Snapshot campaign is incomplete")
    history = json.loads((weights / "training_history.json").read_text(encoding="utf-8"))
    val_loss = {int(row["epoch"]): float(row["val"]["loss"]) for row in history["history"]}
    rows = []
    ledgers = []
    started = perf_counter()
    for checkpoint in snapshots:
        epoch = int(checkpoint.stem.split("_")[-1])
        aggregate, runs = _evaluate_checkpoint(
            repo, checkpoint, targets, [int(value) for value in section["development_source_seeds"]],
            int(section["development_candidates"]), device or section["device"],
            int(section["physics_workers"]),
        )
        row = {"epoch": epoch, "checkpoint": str(checkpoint.relative_to(repo)), "validation_loss": val_loss[epoch], **aggregate}
        rows.append(row)
        for run in runs:
            ledgers.append({"epoch": epoch, **run})
    selected = max(rows, key=_selection_key)
    losses = [row["validation_loss"] for row in rows]
    correlations = {
        "val_loss_vs_hit": float(spearmanr(losses, [row["hit_rate"] for row in rows]).statistic),
        "val_loss_vs_median_rho": float(spearmanr(losses, [row["median_best_rho"] for row in rows]).statistic),
        "val_loss_vs_p90_rho": float(spearmanr(losses, [row["p90_best_rho"] for row in rows]).statistic),
    }
    summary = {
        "protocol_version": VERSION, "status": "COMPLETE", "formal_use_allowed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "lexicographic_hit_p90_median_validation_loss",
        "development_target_count": len(targets), "confirmation_target_count": len(confirmation),
        "snapshots": rows, "selected": selected, "correlations": correlations,
        "elapsed_seconds": perf_counter() - started,
        "hashes": {"confirmation_targets": sha256_file(confirmation_path)},
    }
    _write_json(runtime / "summary.json", summary)
    _write_jsonl(runtime / "development_run_ledger.jsonl", ledgers)
    (runtime / "Checkpoint_Selection_Report.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _confirmation_aggregate(runs: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in runs if row["arm_id"] == arm]
    finite = [row["best_true_rho"] for row in selected if row["best_true_rho"] is not None]
    targets = sorted({row["target_id"] for row in selected})
    hit_counts = {
        target: sum(row["oracle_hit"] for row in selected if row["target_id"] == target)
        for target in targets
    }
    return {
        "runs": len(selected), "targets": len(targets),
        "hit_rate": float(np.mean([row["oracle_hit"] for row in selected])),
        "median_best_rho": float(np.median(finite)),
        "p90_best_rho": float(np.percentile(finite, 90)),
        "mean_physics_acceptable": float(np.mean([row["physics_acceptable"] for row in selected])),
        "any_seed_target_coverage": float(np.mean([value >= 1 for value in hit_counts.values()])),
        "stable_target_fraction": float(np.mean([value >= 2 for value in hit_counts.values()])),
        "all_seed_target_fraction": float(np.mean([value == 3 for value in hit_counts.values()])),
    }


def _render_confirmation(summary: Mapping[str, Any]) -> str:
    lines = [
        "# C下游选择Checkpoint新目标确认",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        f"> C checkpoint由development选定为epoch {summary['selected_epoch']}；确认目标此前未参与选择。",
        "",
        "| Arm | Hit@512 | median rho | P90 rho | any-seed coverage | stable targets | acceptable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("baseline_projection", "c_selected"):
        row = summary["aggregate"][arm]
        lines.append(
            f"| {arm} | {row['hit_rate']:.1%} | {row['median_best_rho']:.3f} | {row['p90_best_rho']:.3f} | "
            f"{row['any_seed_target_coverage']:.1%} | {row['stable_target_fraction']:.1%} | {row['mean_physics_acceptable']:.1f} |"
        )
    paired = summary["paired"]
    lines += [
        "", "## 配对结果", "",
        f"- C新增命中run：{paired['c_wins']}；C丢失命中run：{paired['c_losses']}；状态相同：{paired['ties']}。",
        f"- paired median rho delta (C-baseline)：{paired['median_rho_delta']:+.3f}。",
        f"- 预注册确认门槛：**{summary['gate']}**。", "",
    ]
    return "\n".join(lines)


def run_confirmation_evaluation(root: str | Path, config_path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    repo = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).resolve().read_text(encoding="utf-8"))
    section = config["checkpoint_selection"]
    runtime = repo / section["runtime_root"]
    summary_path = runtime / "summary.json"
    output = runtime / "confirmation_summary.json"
    if output.exists():
        raise ValueError(f"Refusing to overwrite confirmation result: {output}")
    selection = json.loads(summary_path.read_text(encoding="utf-8"))
    targets = _read_jsonl(runtime / "new_locked_confirmation_targets.jsonl")
    selected_checkpoint = repo / selection["selected"]["checkpoint"]
    baseline_checkpoint = repo / section["baseline_checkpoint"]
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    schema = yaml.safe_load((repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8"))
    physics_model = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    seeds = [int(value) for value in section["confirmation_source_seeds"]]
    count = int(section["confirmation_candidates"])
    runs = []
    generators = {
        "baseline_projection": load_generator(baseline_checkpoint, device=device or section["device"]),
        "c_selected": load_generator(selected_checkpoint, device=device or section["device"]),
    }
    with ProcessPoolExecutor(max_workers=int(section["physics_workers"]), initializer=_worker_init, initargs=(str(repo),)) as executor:
        for arm, generator in generators.items():
            for target_row in targets:
                target = Geometry19InverseTarget.from_mapping(target_row["target"])
                physical_condition = np.asarray([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg])
                normalized_condition = torch.as_tensor(generator.normalizer.transform_conditions(physical_condition), dtype=generator.dtype, device=generator.device)
                for seed in seeds:
                    if arm == "baseline_projection":
                        sources = sample_device_invariant_sources(generator, count, seed)
                        with torch.no_grad():
                            samples = integrate_cfm_sources(generator, sources, normalized_condition).cpu().numpy()
                        candidates, tasks, _ = _map_candidates(samples, schema, physics_model, "nearest_integer", target_row["reference_x_19"], count)
                        run_id = f"{arm}:{target.target_id}:{seed}"
                        for task in tasks:
                            task.update({"run_id": run_id, "target": target.as_mapping()})
                    else:
                        samples = generator.sample(normalized_condition, count, seed=seed, space="normalized").values.numpy()
                        candidates, tasks = _filter_samples(samples, arm, target.target_id, seed, schema, physics_model)
                        for task in tasks:
                            task["target"] = target.as_mapping()
                    physics = list(executor.map(_physics_worker, tasks, chunksize=32))
                    by_id = {int(row["candidate_id"]): row for row in physics}
                    rhos = []
                    for row in candidates:
                        result = by_id.get(int(row["candidate_id"]))
                        rho = (result.get("hit_alignment") or {}).get("rho") if result else None
                        if rho is not None:
                            rhos.append(float(rho))
                    runs.append({
                        "arm_id": arm, "target_id": target.target_id,
                        "target_region": target_row["target_region"], "source_seed": seed,
                        "oracle_hit": bool(rhos and min(rhos) <= 1.0),
                        "best_true_rho": min(rhos) if rhos else None,
                        "physics_acceptable": len(rhos),
                    })
    aggregate = {arm: _confirmation_aggregate(runs, arm) for arm in generators}
    base_by = {(row["target_id"], row["source_seed"]): row for row in runs if row["arm_id"] == "baseline_projection"}
    c_by = {(row["target_id"], row["source_seed"]): row for row in runs if row["arm_id"] == "c_selected"}
    keys = sorted(base_by)
    paired = {
        "c_wins": sum(not base_by[key]["oracle_hit"] and c_by[key]["oracle_hit"] for key in keys),
        "c_losses": sum(base_by[key]["oracle_hit"] and not c_by[key]["oracle_hit"] for key in keys),
        "ties": sum(base_by[key]["oracle_hit"] == c_by[key]["oracle_hit"] for key in keys),
        "median_rho_delta": float(np.median([c_by[key]["best_true_rho"] - base_by[key]["best_true_rho"] for key in keys])),
    }
    base, current = aggregate["baseline_projection"], aggregate["c_selected"]
    passed = (
        current["hit_rate"] > base["hit_rate"]
        or (
            current["hit_rate"] == base["hit_rate"]
            and current["p90_best_rho"] < base["p90_best_rho"]
            and current["any_seed_target_coverage"] >= base["any_seed_target_coverage"]
        )
    )
    summary = {
        "protocol_version": VERSION, "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_epoch": int(selection["selected"]["epoch"]),
        "target_count": len(targets), "source_seeds": seeds, "candidates": count,
        "aggregate": aggregate, "paired": paired,
        "gate": "PASS" if passed else "FAIL",
        "hashes": {"confirmation_targets": sha256_file(runtime / "new_locked_confirmation_targets.jsonl")},
    }
    _write_json(output, summary)
    _write_jsonl(runtime / "confirmation_run_ledger.jsonl", runs)
    (runtime / "Confirmation_Report.md").write_text(_render_confirmation(summary), encoding="utf-8")
    return summary
