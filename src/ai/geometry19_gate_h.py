"""Gate H: no-retraining candidate-utilization audit for Geometry19."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.ai.geometry19_candidates import (
    CandidateBatch,
    filter_candidate_batch,
    generate_candidate_batch,
)
from src.ai.geometry19_inverse_contract import load_target, load_wp19_23_contract
from src.ai.geometry19_physics import evaluate_geometry19_candidate
from src.ai.geometry19_pipeline import build_target_mapping, resolve_generator_checkpoint
from src.ai.geometry19_selection import ConstraintDominanceContract
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file
from src.physics import (
    ConstraintProfile,
    Environment,
    Geometry19,
    GeometryStateError,
    ModelConfig,
    derive_geometry_state,
)


GATE_H_VERSION = "wp19_23_gate_h_candidate_utilization_v1"
ARM_IDS = (
    "h0_strict_direct",
    "h1_projected_direct",
    "h2_projected_union",
    "h3_tolerance_ranked_diverse_union",
)


class Geometry19GateHError(ValueError):
    """Raised when the Gate H audit contract is violated."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n"
            )
    temporary.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _design_hash(mapping: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _power_of_two_floor(value: int) -> int:
    if value <= 0:
        return 0
    return 1 << (int(value).bit_length() - 1)


def build_direct_batch(paired: CandidateBatch) -> CandidateBatch:
    """Recover the paired direct arm from a D-Flow generation batch."""

    return CandidateBatch(
        mode="direct",
        candidate_ids=paired.candidate_ids,
        initial_sources=paired.initial_sources,
        direct_normalized=paired.direct_normalized,
        active_normalized=paired.direct_normalized,
        refined_sources=None,
        dflow_history=(),
    )


def project_discrete_candidates(
    batch: CandidateBatch,
    strict_rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    model_config: ModelConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Project integer coordinates to legal grid points and retain unique designs."""

    fields = list(schema["design_variables"])
    names = [item["name"] for item in fields]
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=float)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=float)
    integer_indices = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    if len(strict_rows) != len(batch.candidate_ids):
        raise Geometry19GateHError("Strict ledger and generation batch lengths differ")
    unique: dict[str, dict[str, Any]] = {}
    failures = 0
    repaired_discrete = 0
    for position, candidate_id_value in enumerate(batch.candidate_ids):
        normalized = np.asarray(batch.active_normalized[position], dtype=float)
        if not np.isfinite(normalized).all() or np.any(normalized < -1.0) or np.any(
            normalized > 1.0
        ):
            failures += 1
            continue
        physical = lower + 0.5 * (normalized + 1.0) * (upper - lower)
        distances = np.abs(physical[integer_indices] - np.rint(physical[integer_indices]))
        physical[integer_indices] = np.clip(
            np.rint(physical[integer_indices]), lower[integer_indices], upper[integer_indices]
        )
        mapping = {
            name: int(physical[index]) if index in integer_indices else float(physical[index])
            for index, name in enumerate(names)
        }
        try:
            design = Geometry19.from_mapping(mapping)
            derive_geometry_state(design, model_config)
        except (TypeError, ValueError, GeometryStateError):
            failures += 1
            continue
        canonical = design.as_mapping()
        design_hash = _design_hash(canonical)
        strict = strict_rows[position]
        rescued = strict.get("failure_code") == "INVALID_DISCRETE_COORDINATE"
        repaired_discrete += int(rescued)
        row = {
            "candidate_key": f"{batch.mode}:{int(candidate_id_value)}",
            "candidate_id": int(candidate_id_value),
            "source_modes": [batch.mode],
            "source_candidate_ids": [int(candidate_id_value)],
            "geometry19": canonical,
            "design_hash": design_hash,
            "normalized_geometry": (
                2.0 * (np.asarray([canonical[name] for name in names]) - lower) / (upper - lower)
                - 1.0
            ).tolist(),
            "strict_valid": bool(strict.get("valid")),
            "rescued_discrete": rescued,
            "distance_to_integer": {
                names[index]: float(distance)
                for index, distance in zip(integer_indices, distances, strict=True)
            },
            "predicted_outputs": None,
            "predicted_axis_ratios": None,
            "predicted_rho": None,
        }
        previous = unique.get(design_hash)
        if previous is None or row["candidate_key"] < previous["candidate_key"]:
            unique[design_hash] = row
    rows = sorted(unique.values(), key=lambda row: (row["candidate_id"], row["candidate_key"]))
    return rows, {
        "projected_unique": len(rows),
        "projection_failures": failures,
        "discrete_repaired_rows": repaired_discrete,
        "projection_duplicates": len(batch.candidate_ids) - failures - len(rows),
    }


def merge_candidate_pools(*pools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge candidate pools by canonical design while preserving provenance."""

    merged: dict[str, dict[str, Any]] = {}
    for pool in pools:
        for source in pool:
            design_hash = str(source["design_hash"])
            if design_hash not in merged:
                merged[design_hash] = dict(source)
                continue
            row = merged[design_hash]
            row["source_modes"] = sorted(
                set(row["source_modes"]) | set(source["source_modes"])
            )
            row["source_candidate_ids"] = sorted(
                set(row["source_candidate_ids"]) | set(source["source_candidate_ids"])
            )
            row["strict_valid"] = bool(row["strict_valid"] or source["strict_valid"])
            row["rescued_discrete"] = bool(
                row["rescued_discrete"] or source["rescued_discrete"]
            )
    return sorted(
        merged.values(),
        key=lambda row: (
            min(row["source_candidate_ids"]),
            0 if "direct" in row["source_modes"] else 1,
            row["design_hash"],
        ),
    )


def annotate_tolerance_predictions(
    rows: list[dict[str, Any]],
    target: Mapping[str, Any],
    fields: Sequence[Mapping[str, Any]],
    forward_model: torch.nn.Module,
    normalization: Mapping[str, np.ndarray],
    tolerances: Mapping[str, float],
) -> None:
    """Attach tolerance-normalized forward-surrogate predictions in place."""

    if not rows:
        return
    device = next(forward_model.parameters()).device
    dtype = next(forward_model.parameters()).dtype
    names = [item["name"] for item in fields]
    physical = np.asarray(
        [[row["geometry19"][name] for name in names] for row in rows], dtype=np.float64
    )
    input_mean = np.asarray(normalization["input_mean"], dtype=np.float64)
    input_std = np.asarray(normalization["input_std"], dtype=np.float64)
    normalized = torch.as_tensor(
        (physical - input_mean) / input_std, dtype=dtype, device=device
    )
    with torch.no_grad():
        prediction_normalized = forward_model(normalized)
    output_mean = torch.as_tensor(
        normalization["output_mean"], dtype=dtype, device=device
    )
    output_std = torch.as_tensor(
        normalization["output_std"], dtype=dtype, device=device
    )
    predictions = (prediction_normalized * output_std + output_mean)[:, :3].cpu().numpy()
    output_names = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
    for row, prediction in zip(rows, predictions, strict=True):
        ratios = {
            name: abs(float(value) - float(target[name]))
            / (float(tolerances[name]) * abs(float(target[name])))
            for name, value in zip(output_names, prediction, strict=True)
        }
        row["predicted_outputs"] = {
            name: float(value) for name, value in zip(output_names, prediction, strict=True)
        }
        row["predicted_axis_ratios"] = ratios
        row["predicted_rho"] = max(ratios.values())


def select_tolerance_ranked_diverse(
    rows: Sequence[Mapping[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Use half the budget for predicted rho and half for deterministic diversity."""

    if budget <= 0 or budget > len(rows):
        raise Geometry19GateHError("Invalid ranked-diverse selection budget")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (float(row["predicted_rho"]), row["candidate_key"]),
    )
    exploitation_count = max(1, budget // 2)
    selected = ordered[:exploitation_count]
    selected_hashes = {row["design_hash"] for row in selected}
    shortlist = [
        row for row in ordered[: min(len(ordered), max(4 * budget, budget))]
        if row["design_hash"] not in selected_hashes
    ]
    while len(selected) < budget:
        matrix = np.asarray([row["normalized_geometry"] for row in selected], dtype=float)
        best_index = max(
            range(len(shortlist)),
            key=lambda index: (
                float(
                    np.linalg.norm(
                        matrix - np.asarray(shortlist[index]["normalized_geometry"], dtype=float),
                        axis=1,
                    ).min()
                ),
                -float(shortlist[index]["predicted_rho"]),
                shortlist[index]["candidate_key"],
            ),
        )
        selected.append(shortlist.pop(best_index))
    return selected


def _strict_pool(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for row in rows:
        if not row.get("valid"):
            continue
        values.append(
            {
                "candidate_key": f"direct:{int(row['candidate_id'])}",
                "candidate_id": int(row["candidate_id"]),
                "source_modes": ["direct"],
                "source_candidate_ids": [int(row["candidate_id"])],
                "geometry19": dict(row["geometry19"]),
                "design_hash": str(row["design_hash"]),
                "normalized_geometry": list(row["normalized_design"]),
                "strict_valid": True,
                "rescued_discrete": False,
                "predicted_outputs": None,
                "predicted_axis_ratios": None,
                "predicted_rho": None,
            }
        )
    return values


def _true_rho(evaluation: Mapping[str, Any], tolerances: Mapping[str, float]) -> float | None:
    errors = evaluation.get("relative_errors")
    if not evaluation.get("physics_acceptable") or not isinstance(errors, Mapping):
        return None
    return max(float(errors[name]) / float(tolerances[name]) for name in tolerances)


def _arm_metrics(
    selected: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    rho_values = [
        value for value in (_true_rho(row, tolerances) for row in evaluations) if value is not None
    ]
    hits = [row for row in evaluations if row["development_diagnostic_hit"]]
    return {
        "selected": len(selected),
        "physics_acceptable": sum(bool(row["physics_acceptable"]) for row in evaluations),
        "diagnostic_hits": len(hits),
        "target_hit": bool(hits),
        "best_true_rho": min(rho_values) if rho_values else None,
        "median_true_rho": float(np.median(rho_values)) if rho_values else None,
        "best_total_true_error": min(
            (
                float(row["total_true_error"])
                for row in evaluations
                if row.get("physics_acceptable") and row.get("total_true_error") is not None
            ),
            default=None,
        ),
        "source_mode_counts": {
            mode: sum(mode in row["source_modes"] for row in selected)
            for mode in ("direct", "dflow")
        },
        "discrete_rescued_selected": sum(bool(row["rescued_discrete"]) for row in selected),
        "selected_design_hashes": [row["design_hash"] for row in selected],
    }


def render_gate_h_report(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    labels = {
        "h0_strict_direct": "H0 strict direct",
        "h1_projected_direct": "H1 projected direct",
        "h2_projected_union": "H2 projected union",
        "h3_tolerance_ranked_diverse_union": "H3 tolerance-ranked + diverse union",
    }
    lines = [
        "# WP19-23 Gate H：不重训候选利用率验证",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> 状态：开发级固定 V2.1 预算实验；不授权生产协议或默认模型替换。",
        "",
        "## 比较口径",
        "",
        f"- {summary['target_count']} 个 held-out 已知可达 ID 目标；",
        f"- 每个目标生成 {summary['raw_candidates_per_mode']} 个 paired direct/D-Flow 候选；",
        "- 所有臂逐目标共享相同 V2.1 逻辑预算；",
        "- H1/H2/H3 仅做最近整数投影，不重训生成器；",
        "- H3 以预测 `rho=max(e_i/tau_i)` 排序，半数预算保留给设计多样性。",
        "",
        "## 汇总",
        "",
        "| 实验臂 | 目标命中 | 逻辑V2.1 | 物理有效 | median最佳rho | 投影救援入选 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_IDS:
        row = aggregate[arm]
        rho = "N/A" if row["median_best_true_rho"] is None else f"{row['median_best_true_rho']:.4g}"
        lines.append(
            f"| {labels[arm]} | {row['target_hits']}/{row['targets']} | "
            f"{row['logical_v2_1_evaluations']} | {row['physics_acceptable']} | {rho} | "
            f"{row['discrete_rescued_selected']} |"
        )
    lines.extend(
        [
            "",
            "命中判据保持三轴 AND 不变；`rho <= 1` 等价于三轴全部通过。",
            "",
            "## Gate H 判定",
            "",
            f"- 运行状态：**{summary['gate_h']['development_status']}**；",
            f"- 等预算：**{summary['gate_h']['equal_budget_status']}**；",
            f"- 最佳开发臂：**{summary['gate_h']['best_arm_by_hits_then_rho']}**；",
            "- Formal Gate：**DEFERRED**。",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate_h_campaign(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the four-arm Gate H candidate-utilization campaign."""

    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != GATE_H_VERSION:
        raise Geometry19GateHError("Gate H protocol version mismatch")
    contract_path = root / "config/physics/wp19_23_inverse_closure_v1.yaml"
    contract = load_wp19_23_contract(contract_path)
    protocol = contract.payload
    raw_count = int(protocol["generation"]["raw_candidate_count"])
    if int(config["budgets"]["raw_candidates_per_target_per_mode"]) != raw_count:
        raise Geometry19GateHError("Gate H raw count differs from production contract")
    runtime = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / config["artifacts"]["runtime_root"]).resolve()
    )
    if runtime.exists():
        raise Geometry19GateHError(f"Refusing to overwrite Gate H run: {runtime}")
    runtime.mkdir(parents=True)
    checkpoint = resolve_generator_checkpoint(root, "bounded-sigmoid")
    runtime_device = device or str(config["device"])
    schema_path = root / protocol["contracts"]["design_schema"]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    fields = list(schema["design_variables"])
    model_config = ModelConfig.from_yaml(root / protocol["contracts"]["calibration_pack"])
    constraint_profile = ConstraintProfile.from_yaml(
        root / protocol["contracts"]["constraint_profile"]
    )
    dominance = ConstraintDominanceContract.from_yaml(
        root / protocol["contracts"]["constraint_dominance"]
    )
    forward_contract = protocol["models"]["forward"]
    forward_root = root / forward_contract["variants"]["s0"]["weights"]
    forward_manifest = json.loads(
        (forward_root / "model_manifest.json").read_text(encoding="utf-8")
    )
    forward, forward_stats, _ = load_weight_bundle(
        forward_root,
        expected_model_tag=forward_manifest["model_tag"],
        expected_feature_names=forward_manifest["feature_names"],
        expected_target_names=forward_manifest["target_names"],
        device=runtime_device,
    )
    forward.eval().requires_grad_(False)
    tolerances = {
        name: float(value)
        for name, value in protocol["diagnostic_acceptance"]["relative_tolerances"].items()
    }
    target_runs: list[dict[str, Any]] = []
    global_unique_evaluations = 0
    for index, target_spec in enumerate(config["targets"]):
        target_mapping = build_target_mapping(
            thrust_N=target_spec["thrust_N"],
            tsfc_kg_N_s=target_spec["tsfc_kg_N_s"],
            mass_kg=target_spec["mass_kg"],
            target_type=target_spec["target_type"],
            expected_status=target_spec["expected_status"],
        )
        target_runtime = runtime / f"target_{index:02d}_{target_mapping['target_id'][:12]}"
        target_path = target_runtime / "target.yaml"
        _write_yaml(target_path, target_mapping)
        target = load_target(target_path)
        started = perf_counter()
        paired = generate_candidate_batch(
            root,
            target,
            mode="dflow",
            device=runtime_device,
            generator_checkpoint=checkpoint,
            forward_model_tag="s0",
        )
        direct = build_direct_batch(paired)
        strict_tolerance = float(
            protocol["candidate_filter"]["discrete_absolute_tolerance_count"]
        )
        direct_strict_rows, direct_strict_summary = filter_candidate_batch(
            direct, schema, model_config, discrete_tolerance_count=strict_tolerance
        )
        dflow_strict_rows, dflow_strict_summary = filter_candidate_batch(
            paired, schema, model_config, discrete_tolerance_count=strict_tolerance
        )
        strict_direct = _strict_pool(direct_strict_rows)
        projected_direct, projected_direct_counts = project_discrete_candidates(
            direct, direct_strict_rows, schema, model_config
        )
        projected_dflow, projected_dflow_counts = project_discrete_candidates(
            paired, dflow_strict_rows, schema, model_config
        )
        union = merge_candidate_pools(projected_direct, projected_dflow)
        annotate_tolerance_predictions(
            union,
            target_mapping,
            fields,
            forward,
            forward_stats,
            tolerances,
        )
        prediction_by_hash = {row["design_hash"]: row for row in union}
        for pool in (strict_direct, projected_direct):
            for row in pool:
                prediction = prediction_by_hash.get(row["design_hash"])
                if prediction is not None:
                    row["predicted_outputs"] = prediction["predicted_outputs"]
                    row["predicted_axis_ratios"] = prediction["predicted_axis_ratios"]
                    row["predicted_rho"] = prediction["predicted_rho"]
        cap = int(config["budgets"]["maximum_true_physics_per_target_per_arm"])
        available = min(len(strict_direct), len(projected_direct), len(union), cap)
        budget = _power_of_two_floor(available)
        if budget <= 0:
            raise Geometry19GateHError(f"No common Gate H budget for target {target.target_id}")
        selected_by_arm = {
            "h0_strict_direct": strict_direct[:budget],
            "h1_projected_direct": projected_direct[:budget],
            "h2_projected_union": union[:budget],
            "h3_tolerance_ranked_diverse_union": select_tolerance_ranked_diverse(
                union, budget
            ),
        }
        cache: dict[str, dict[str, Any]] = {}
        evaluations_by_arm: dict[str, list[dict[str, Any]]] = {}
        environment = Environment()
        for arm, selected in selected_by_arm.items():
            evaluations = []
            for position, row in enumerate(selected):
                design_hash = str(row["design_hash"])
                if design_hash not in cache:
                    cache[design_hash] = evaluate_geometry19_candidate(
                        position,
                        arm,
                        row["geometry19"],
                        design_hash,
                        target,
                        contract.diagnostic_acceptance,
                        model_config,
                        constraint_profile,
                        environment,
                        constraint_dominance_contract=dominance,
                    )
                evaluation = dict(cache[design_hash])
                evaluation["candidate_id"] = position
                evaluation["evaluation_id"] = f"{arm}:{position}"
                evaluation["variant"] = arm
                evaluation["predicted_rho"] = row["predicted_rho"]
                evaluations.append(evaluation)
            evaluations_by_arm[arm] = evaluations
        global_unique_evaluations += len(cache)
        arms = {
            arm: {
                "pool_size": {
                    "h0_strict_direct": len(strict_direct),
                    "h1_projected_direct": len(projected_direct),
                    "h2_projected_union": len(union),
                    "h3_tolerance_ranked_diverse_union": len(union),
                }[arm],
                **_arm_metrics(selected_by_arm[arm], evaluations_by_arm[arm], tolerances),
            }
            for arm in ARM_IDS
        }
        for arm in ARM_IDS:
            arm_root = target_runtime / arm
            _write_jsonl(arm_root / "selected_candidates.jsonl", selected_by_arm[arm])
            _write_jsonl(arm_root / "physics_evaluations.jsonl", evaluations_by_arm[arm])
        target_summary = {
            "target_index": index,
            "dataset_sample_id": int(target_spec["sample_id"]),
            "target": target_mapping,
            "common_budget": budget,
            "generation_elapsed_seconds": perf_counter() - started,
            "candidate_funnel": {
                "raw_per_mode": raw_count,
                "strict_direct": direct_strict_summary,
                "strict_dflow": dflow_strict_summary,
                "projected_direct": projected_direct_counts,
                "projected_dflow": projected_dflow_counts,
                "projected_union_unique": len(union),
            },
            "unique_v2_1_evaluations_with_cross_arm_cache": len(cache),
            "arms": arms,
        }
        _write_json(target_runtime / "target_summary.json", target_summary)
        _write_jsonl(target_runtime / "projected_union.jsonl", union)
        target_runs.append(target_summary)
    aggregate: dict[str, Any] = {}
    for arm in ARM_IDS:
        arm_rows = [run["arms"][arm] for run in target_runs]
        best_rho = [row["best_true_rho"] for row in arm_rows if row["best_true_rho"] is not None]
        aggregate[arm] = {
            "targets": len(arm_rows),
            "target_hits": sum(row["target_hit"] for row in arm_rows),
            "logical_v2_1_evaluations": sum(row["selected"] for row in arm_rows),
            "physics_acceptable": sum(row["physics_acceptable"] for row in arm_rows),
            "median_pool_size": float(np.median([row["pool_size"] for row in arm_rows])),
            "median_best_true_rho": float(np.median(best_rho)) if best_rho else None,
            "discrete_rescued_selected": sum(
                row["discrete_rescued_selected"] for row in arm_rows
            ),
        }
    best_arm = min(
        ARM_IDS,
        key=lambda arm: (
            -aggregate[arm]["target_hits"],
            aggregate[arm]["median_best_true_rho"]
            if aggregate[arm]["median_best_true_rho"] is not None
            else math.inf,
            ARM_IDS.index(arm),
        ),
    )
    equal_budget = all(
        len({run["arms"][arm]["selected"] for arm in ARM_IDS}) == 1
        for run in target_runs
    )
    summary = {
        "protocol_version": GATE_H_VERSION,
        "work_package": "WP19-23-Gate-H",
        "status": "PASS" if equal_budget else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "target_count": len(target_runs),
        "raw_candidates_per_mode": raw_count,
        "generator_checkpoint": str(checkpoint),
        "generator_checkpoint_sha256": sha256_file(checkpoint),
        "target_runs": target_runs,
        "aggregate": aggregate,
        "actual_unique_v2_1_evaluations_with_cross_arm_cache": global_unique_evaluations,
        "gate_h": {
            "development_status": "PASS" if equal_budget else "FAIL",
            "equal_budget_status": "PASS" if equal_budget else "FAIL",
            "best_arm_by_hits_then_rho": best_arm,
            "formal_gate_status": "DEFERRED",
            "default_model_replacement": "NO_ACTION",
        },
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "contract_sha256": sha256_file(contract_path),
            "schema_sha256": sha256_file(schema_path),
            "code_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
    }
    summary_path = runtime / config["artifacts"]["summary"]
    report_path = runtime / config["artifacts"]["report"]
    _write_json(summary_path, summary)
    report_path.write_text(render_gate_h_report(summary), encoding="utf-8")
    manifest_path = runtime / config["artifacts"]["manifest"]
    artifacts = [
        {
            "path": str(path.relative_to(runtime)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(path for path in runtime.rglob("*") if path.is_file())
        if path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "manifest_version": "wp19_23_gate_h_manifest_v1",
            "summary_sha256": sha256_file(summary_path),
            "report_sha256": sha256_file(report_path),
            "artifacts": artifacts,
        },
    )
    return summary
