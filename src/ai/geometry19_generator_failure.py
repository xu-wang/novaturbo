"""Failure decomposition for the frozen Geometry19 bounded generator."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch
import yaml

from src.ai.generators import CFMGenerator, load_generator
from src.ai.geometry19_candidates import sample_device_invariant_sources
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_generator_oracle import _physics_worker, _read_jsonl, _worker_init, _write_json, _write_jsonl
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.ai.geometry19_surrogate import sha256_file
from src.physics import Geometry19, GeometryStateError, ModelConfig, derive_geometry_state


VERSION = "wp19_23_generator_failure_decomposition_v1"
INTEGER_NAMES = ("compressor_blade_count", "combustor_injector_count", "turbine_blade_count")


class GeneratorFailureError(ValueError):
    pass


def _design_hash(mapping: Mapping[str, Any]) -> str:
    return sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _select_targets(suite: Sequence[Mapping[str, Any]], per_region: int) -> list[dict[str, Any]]:
    train = [dict(row) for row in suite if row["split"] == "train_diagnostic"]
    held: list[dict[str, Any]] = []
    for region in ("dense", "mid", "sparse"):
        region_rows = [
            dict(row)
            for row in suite
            if row["split"] == "locked_held_out" and row["target_region"] == region
        ]
        held.extend(region_rows[:per_region])
    return train + held


def _map_candidates(
    normalized: np.ndarray,
    schema: Mapping[str, Any],
    model_config: ModelConfig,
    mode: str,
    reference: Mapping[str, Any],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fields = list(schema["design_variables"])
    names = [item["name"] for item in fields]
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=float)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=float)
    integer = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    reference_vector = np.asarray([reference[name] for name in names], dtype=float)
    reference_normalized = 2.0 * (reference_vector - lower) / (upper - lower) - 1.0
    compact: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    mapped_vectors: list[np.ndarray] = []
    seen: set[str] = set()
    for candidate_id, vector in enumerate(normalized[:limit]):
        failure = None
        physical = lower + 0.5 * (vector + 1.0) * (upper - lower)
        if not np.isfinite(vector).all():
            failure = "NON_FINITE_GENERATION"
        elif np.any(vector < -1.0) or np.any(vector > 1.0):
            failure = "OUT_OF_SCHEMA_BOUNDS"
        else:
            rounded = np.rint(physical[integer])
            if mode == "strict" and np.any(np.abs(physical[integer] - rounded) > 0.25 + 1e-12):
                failure = "INVALID_DISCRETE_COORDINATE"
            elif mode == "reference_discrete_oracle":
                physical[integer] = reference_vector[integer]
            else:
                physical[integer] = rounded
        geometry = None
        design_hash = None
        geometry_l2 = None
        if failure is None:
            mapping = {name: int(physical[index]) if index in integer else float(physical[index]) for index, name in enumerate(names)}
            try:
                design = Geometry19.from_mapping(mapping)
                derive_geometry_state(design, model_config)
                geometry = design.as_mapping()
                design_hash = _design_hash(geometry)
                if design_hash in seen:
                    failure = "DUPLICATE_CANDIDATE"
                else:
                    seen.add(design_hash)
                    mapped = 2.0 * (np.asarray([geometry[name] for name in names]) - lower) / (upper - lower) - 1.0
                    geometry_l2 = float(np.linalg.norm(mapped - reference_normalized))
                    mapped_vectors.append(mapped)
            except (TypeError, ValueError, GeometryStateError):
                failure = "GEOMETRY_MAPPING_FAILED"
        row = {"candidate_id": candidate_id, "mode": mode, "failure_code": failure, "design_hash": design_hash, "geometry_l2_to_reference": geometry_l2}
        compact.append(row)
        if failure is None:
            tasks.append({"candidate_id": candidate_id, "geometry19": geometry, "design_hash": design_hash})
    matrix = np.asarray(mapped_vectors, dtype=float)
    if len(matrix) >= 2:
        covariance = np.cov(matrix, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        positive = eigenvalues[eigenvalues > 1e-12]
        effective_rank = float(np.exp(-np.sum((positive / positive.sum()) * np.log(positive / positive.sum())))) if len(positive) else 0.0
        dimension_std = np.std(matrix, axis=0)
        diversity = {"mapped": len(matrix), "effective_rank": effective_rank, "median_dimension_std": float(np.median(dimension_std)), "minimum_dimension_std": float(np.min(dimension_std)), "maximum_dimension_std": float(np.max(dimension_std))}
    else:
        diversity = {"mapped": len(matrix), "effective_rank": None, "median_dimension_std": None, "minimum_dimension_std": None, "maximum_dimension_std": None}
    return compact, tasks, diversity


def _attach_physics(compact: list[dict[str, Any]], physics: Sequence[Mapping[str, Any]]) -> None:
    by_id = {int(row["candidate_id"]): row for row in physics}
    for row in compact:
        value = by_id.get(int(row["candidate_id"]))
        row["physics_acceptable"] = bool(value and value["physics_acceptable"])
        row["rho"] = (value.get("hit_alignment") or {}).get("rho") if value else None


def _metrics(rows: Sequence[Mapping[str, Any]], prefixes: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix in prefixes:
        subset = rows[:prefix]
        rhos = [float(row["rho"]) for row in subset if row.get("rho") is not None]
        best_row = min((row for row in subset if row.get("rho") is not None), key=lambda row: row["rho"], default=None)
        nearest_row = min((row for row in subset if row.get("rho") is not None and row.get("geometry_l2_to_reference") is not None), key=lambda row: row["geometry_l2_to_reference"], default=None)
        pairs = [(float(row["geometry_l2_to_reference"]), float(row["rho"])) for row in subset if row.get("rho") is not None and row.get("geometry_l2_to_reference") is not None]
        correlation = float(spearmanr(*zip(*pairs)).statistic) if len(pairs) >= 3 else None
        result[str(prefix)] = {
            "oracle_hit": bool(rhos and min(rhos) <= 1.0), "best_true_rho": min(rhos) if rhos else None,
            "strict_hits": sum(value <= 1.0 for value in rhos), "near_hits_2": sum(value <= 2.0 for value in rhos),
            "physics_acceptable": len(rhos), "prefilter_valid": sum(row["failure_code"] is None for row in subset),
            "best_candidate_geometry_l2": best_row.get("geometry_l2_to_reference") if best_row else None,
            "nearest_geometry_rho": nearest_row.get("rho") if nearest_row else None,
            "geometry_rho_spearman": correlation,
        }
    return result


def _aggregate(runs: Sequence[Mapping[str, Any]], split: str, mode: str, prefixes: Sequence[int]) -> dict[str, Any]:
    selected = [row for row in runs if row["split"] == split and row["mode"] == mode]
    output: dict[str, Any] = {"runs": len(selected), "targets": len({row["target_id"] for row in selected})}
    for prefix in prefixes:
        rows = [run["metrics"][str(prefix)] for run in selected]
        finite = [row["best_true_rho"] for row in rows if row["best_true_rho"] is not None]
        correlations = [row["geometry_rho_spearman"] for row in rows if row["geometry_rho_spearman"] is not None and math.isfinite(row["geometry_rho_spearman"])]
        output[str(prefix)] = {
            "run_hit_rate": float(np.mean([row["oracle_hit"] for row in rows])),
            "median_best_rho": float(np.median(finite)) if finite else None,
            "p90_best_rho": float(np.percentile(finite, 90)) if finite else None,
            "mean_prefilter_valid": float(np.mean([row["prefilter_valid"] for row in rows])),
            "mean_physics_acceptable": float(np.mean([row["physics_acceptable"] for row in rows])),
            "median_geometry_rho_spearman": float(np.median(correlations)) if correlations else None,
            "median_nearest_geometry_rho": float(np.median([row["nearest_geometry_rho"] for row in rows if row["nearest_geometry_rho"] is not None])),
        }
    return output


def _target_coverage(runs: Sequence[Mapping[str, Any]], split: str, mode: str, prefix: int) -> dict[str, float]:
    selected = [row for row in runs if row["split"] == split and row["mode"] == mode]
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["target_id"]] = counts.get(row["target_id"], 0) + int(row["metrics"][str(prefix)]["oracle_hit"])
    return {"any_seed": sum(value >= 1 for value in counts.values()) / len(counts), "two_of_three": sum(value >= 2 for value in counts.values()) / len(counts), "all_three": sum(value == 3 for value in counts.values()) / len(counts)}


def _render(summary: Mapping[str, Any]) -> str:
    def metric(value: Any, spec: str = ".3f") -> str:
        return "N/A" if value is None else format(float(value), spec)

    lines = ["# Geometry19 生成模型失败原因分解", "", f"> 生成时间：{summary['generated_at_utc']}", "> 只做冻结 bounded-sigmoid 生成器分析；未训练或测试任何改进模型。", "", "## F1/F2：训练可逆性与 Best-of-N", "", "| Split | N | 运行命中率 | 任一seed目标覆盖 | median best rho | P90 best rho |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("train_diagnostic", "locked_held_out"):
        for prefix in summary["prefixes"]:
            row = summary["aggregate"]["strict"][split][str(prefix)]
            coverage = summary["target_coverage"][split][str(prefix)]
            lines.append(f"| {split} | {prefix} | {row['run_hit_rate']:.1%} | {coverage['any_seed']:.1%} | {row['median_best_rho']:.3f} | {row['p90_best_rho']:.3f} |")
    lines += ["", "## F4：离散变量诊断（N=512）", "", "| Mode | Split | 运行命中率 | median best rho | mean prefilter valid | mean V2.1 acceptable |", "|---|---|---:|---:|---:|---:|"]
    diagnostic_key = str(summary["diagnostic_candidates"])
    for mode in ("strict","nearest_integer","reference_discrete_oracle"):
        for split in ("train_diagnostic","locked_held_out"):
            row=summary["aggregate"][mode][split][diagnostic_key]
            lines.append(f"| {mode} | {split} | {row['run_hit_rate']:.1%} | {row['median_best_rho']:.3f} | {row['mean_prefilter_valid']:.1f} | {row['mean_physics_acceptable']:.1f} |")
    relation = summary["aggregate"]["strict"]["locked_held_out"][diagnostic_key]
    lines += ["", "## F3/F5：多样性与几何—性能关系", "", f"- strict 生成分布 median effective rank：{metric(summary['diversity']['strict_median_effective_rank'], '.2f')} / 19；", f"- strict median per-dimension std：{metric(summary['diversity']['strict_median_dimension_std'])}；", f"- held-out geometry-distance / true-rho median Spearman：{metric(relation['median_geometry_rho_spearman'])}；", f"- held-out 最近参考几何候选的 median true rho：{metric(relation['median_nearest_geometry_rho'])}。", "", "## 原因判定", ""]
    for value in summary["conclusions"]:
        lines.append(f"- {value}")
    lines += ["", "实验到此停止；未开展网络结构、loss、mixed head 或框架增强实验。", ""]
    return "\n".join(lines)


def run_failure_decomposition(root: str | Path, config_path: str | Path, *, device: str | None = None, output_root: str | Path | None = None, smoke: bool = False) -> dict[str, Any]:
    repo = Path(root).resolve(); source = Path(config_path).resolve(); config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config["protocol_version"] != VERSION: raise GeneratorFailureError("Protocol mismatch")
    runtime = Path(output_root).resolve() if output_root else repo / config["artifacts"]["runtime_root"]
    if runtime.exists(): raise GeneratorFailureError(f"Refusing to overwrite {runtime}")
    runtime.mkdir(parents=True)
    suite = _read_jsonl(repo / config["source_oracle_runtime"] / "target_suite.jsonl")
    targets = _select_targets(suite, 1 if smoke else int(config["targets"]["locked_held_out_per_region"]))
    if smoke: targets = targets[:1] + [row for row in targets if row["split"] == "locked_held_out"][:1]
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    schema = yaml.safe_load((repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8"))
    model_config = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    checkpoint = repo / config["generator_checkpoint"]
    generator = load_generator(checkpoint, device=device or config["execution"]["device"])
    if not isinstance(generator, CFMGenerator): raise GeneratorFailureError("Generator is not CFM")
    maximum = 64 if smoke else int(config["sampling"]["maximum_raw_candidates"])
    prefixes = [64] if smoke else [int(value) for value in config["sampling"]["nested_prefixes"]]
    diagnostic_n = 64 if smoke else int(config["sampling"]["discrete_diagnostic_candidates"])
    seeds = [int(config["sampling"]["source_seeds"][0])] if smoke else [int(value) for value in config["sampling"]["source_seeds"]]
    run_rows=[]; compact_all=[]; diversity_rows=[]; physics_calls=0; started=perf_counter()
    with ProcessPoolExecutor(max_workers=2 if smoke else int(config["execution"]["physics_workers"]), initializer=_worker_init, initargs=(str(repo),)) as executor:
        for target_row in targets:
            target=Geometry19InverseTarget.from_mapping(target_row["target"]); condition=np.asarray([target.thrust_N,target.tsfc_kg_N_s,target.mass_kg])
            normalized_condition=torch.as_tensor(generator.normalizer.transform_conditions(condition),dtype=generator.dtype,device=generator.device)
            for seed in seeds:
                sources=sample_device_invariant_sources(generator,maximum,seed)
                with torch.no_grad(): samples=integrate_cfm_sources(generator,sources,normalized_condition).cpu().numpy()
                for mode in ("strict","nearest_integer","reference_discrete_oracle"):
                    limit=maximum if mode=="strict" else diagnostic_n
                    compact,tasks,diversity=_map_candidates(samples,schema,model_config,mode,target_row["reference_x_19"],limit)
                    run_id=f"{target.target_id}:{seed}:{mode}"
                    for task in tasks: task.update({"run_id":run_id,"target":target.as_mapping()})
                    physics=list(executor.map(_physics_worker,tasks,chunksize=int(config["execution"]["physics_chunksize"])))
                    physics_calls += len(tasks); _attach_physics(compact,physics)
                    for row in compact: row.update({"run_id":run_id,"target_id":target.target_id,"split":target_row["split"],"target_region":target_row["target_region"],"source_seed":seed})
                    compact_all.extend(compact)
                    active_prefixes=prefixes if mode=="strict" else [diagnostic_n]
                    run_rows.append({"run_id":run_id,"target_id":target.target_id,"split":target_row["split"],"target_region":target_row["target_region"],"source_seed":seed,"mode":mode,"metrics":_metrics(compact,active_prefixes),"diversity":diversity})
                    diversity_rows.append({"mode":mode,**diversity})
    modes=("strict","nearest_integer","reference_discrete_oracle")
    aggregate={mode:{split:_aggregate(run_rows,split,mode,prefixes if mode=="strict" else [diagnostic_n]) for split in ("train_diagnostic","locked_held_out")} for mode in modes}
    coverage={split:{str(prefix):_target_coverage(run_rows,split,"strict",prefix) for prefix in prefixes} for split in ("train_diagnostic","locked_held_out")}
    strict_div=[row for row in diversity_rows if row["mode"]=="strict" and row["effective_rank"] is not None]
    diversity={"strict_median_effective_rank":float(np.median([row["effective_rank"] for row in strict_div])),"strict_median_dimension_std":float(np.median([row["median_dimension_std"] for row in strict_div]))}
    train512=aggregate["strict"]["train_diagnostic"][str(prefixes[0])]; held512=aggregate["strict"]["locked_held_out"][str(prefixes[0])]; held8192=aggregate["strict"]["locked_held_out"][str(prefixes[-1])]; rounded=aggregate["nearest_integer"]["locked_held_out"][str(diagnostic_n)]; oracle=aggregate["reference_discrete_oracle"]["locked_held_out"][str(diagnostic_n)]
    conclusions=[
        f"训练目标在 N={prefixes[0]} 的运行命中率仅 {train512['run_hit_rate']:.1%}、median best rho={train512['median_best_rho']:.3f}，与 held-out 的 {held512['run_hit_rate']:.1%}/{held512['median_best_rho']:.3f} 同量级；失败首先是拟合/条件集中能力不足，不是单纯 held-out 泛化。",
        f"held-out 从 N={prefixes[0]} 扩到 {prefixes[-1]} 后运行命中率由 {held512['run_hit_rate']:.1%} 增至 {held8192['run_hit_rate']:.1%}，目标任一 seed 覆盖为 {coverage['locked_held_out'][str(prefixes[-1])]['any_seed']:.1%}；据曲线判断采样量能缓解但不能替代生成分布改进。",
        f"N={diagnostic_n} 最近整数映射 held-out 命中率 {rounded['run_hit_rate']:.1%}，参考离散变量 oracle 为 {oracle['run_hit_rate']:.1%}，相对 strict 的 {held512['run_hit_rate']:.1%} 用于量化 mixed-variable 表示贡献。",
        f"strict 有效输出协方差的 median effective rank={diversity['strict_median_effective_rank']:.2f}/19；是否存在严重 mode collapse 需结合 Best-of-N 饱和和逐维方差共同判断。",
    ]
    summary={"protocol_version":VERSION,"status":"SMOKE_PASS" if smoke else "COMPLETE_ANALYSIS_ONLY","formal_use_allowed":False,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"target_count":len(targets),"source_seeds":seeds,"maximum_raw_candidates":maximum,"prefixes":prefixes,"diagnostic_candidates":diagnostic_n,"physics_calls":physics_calls,"elapsed_seconds":perf_counter()-started,"aggregate":aggregate,"target_coverage":coverage,"diversity":diversity,"conclusions":conclusions,"excluded":["new_training","architecture_change","loss_change","dflow","reranker","gradient_refinement","nsga2"],"hashes":{"config":sha256_file(source),"checkpoint":sha256_file(checkpoint)}}
    _write_jsonl(runtime/config["artifacts"]["run_ledger"],run_rows); _write_jsonl(runtime/config["artifacts"]["candidate_ledger"],compact_all); _write_json(runtime/config["artifacts"]["summary"],summary); (runtime/config["artifacts"]["report"]).write_text(_render(summary),encoding="utf-8")
    return summary
