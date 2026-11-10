"""Target-conditioned pairwise reranker for frozen Geometry19 candidate ledgers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from src.ai.geometry19_performance_surrogate_v2 import _load_model, _predict
from src.ai.geometry19_training import PERFORMANCE_NAMES
from src.physics import Geometry19


PROTOCOL_VERSION = "wp19_23_independent_reranker_v1"


class Geometry19RerankerError(ValueError):
    """Raised when the reranker experiment contract is invalid."""


class PairwiseReranker(nn.Module):
    """Small scalar scorer; larger scores mean an earlier shortlist position."""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_layers:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU(), nn.LayerNorm(int(hidden))))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def true_utility(rho: float | None, run_worst_rho: float) -> float:
    """Map truth to a ranking utility without assigning fake rho to invalid rows."""
    if rho is None or not math.isfinite(float(rho)):
        return -math.log1p(run_worst_rho) - 1.0
    return -math.log1p(max(float(rho), 0.0))


def ranking_metrics(
    rows: Sequence[Mapping[str, Any]], scores: np.ndarray, budgets: Sequence[int]
) -> dict[str, Any]:
    """Evaluate shortlist success per production-like run and per target."""
    if len(rows) != len(scores):
        raise Geometry19RerankerError("rows and scores must have equal length")
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["run_id"]), []).append(index)
    run_results: list[dict[str, Any]] = []
    for run_id, indices in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), index))
        hit_positions = [
            rank
            for rank, index in enumerate(ordered, start=1)
            if rows[index].get("rho") is not None and float(rows[index]["rho"]) <= 1.0
        ]
        run_results.append(
            {
                "run_id": run_id,
                "target_id": str(rows[indices[0]]["target_id"]),
                "candidate_count": len(indices),
                "has_true_hit": bool(hit_positions),
                "first_hit_rank": hit_positions[0] if hit_positions else None,
                "success": {str(k): bool(hit_positions and hit_positions[0] <= k) for k in budgets},
            }
        )
    hit_runs = [row for row in run_results if row["has_true_hit"]]
    hittable_targets = sorted({row["target_id"] for row in hit_runs})
    return {
        "run_count": len(run_results),
        "hit_bearing_run_count": len(hit_runs),
        "hittable_target_count": len(hittable_targets),
        "all_run_success": {
            str(k): float(np.mean([row["success"][str(k)] for row in run_results]))
            if run_results
            else 0.0
            for k in budgets
        },
        "hit_bearing_recall": {
            str(k): float(np.mean([row["success"][str(k)] for row in hit_runs]))
            if hit_runs
            else 0.0
            for k in budgets
        },
        "target_recall": {
            str(k): (
                sum(
                    any(
                        row["target_id"] == target_id and row["success"][str(k)]
                        for row in hit_runs
                    )
                    for target_id in hittable_targets
                )
                / len(hittable_targets)
                if hittable_targets
                else 0.0
            )
            for k in budgets
        },
        "median_first_hit_rank": (
            float(np.median([row["first_hit_rank"] for row in hit_runs])) if hit_runs else None
        ),
        "runs": run_results,
    }


def build_pair_indices(
    rows: Sequence[Mapping[str, Any]],
    utilities: np.ndarray,
    *,
    seed: int,
    pairs_per_run: int,
    elite_per_run: int,
    minimum_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic within-run RankNet pairs, emphasizing the useful head."""
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["run_id"]), []).append(index)
    good: list[int] = []
    bad: list[int] = []
    weights: list[float] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda index: (-float(utilities[index]), index))
        elite = ordered[: min(elite_per_run, len(ordered) - 1)]
        for pair_number in range(pairs_per_run):
            if pair_number < pairs_per_run // 2 and elite:
                left = int(rng.choice(elite))
                worse = [i for i in indices if utilities[i] < utilities[left] - minimum_gap]
                if not worse:
                    continue
                right = int(rng.choice(worse))
            else:
                left, right = rng.choice(indices, size=2, replace=False).tolist()
                if abs(float(utilities[left] - utilities[right])) < minimum_gap:
                    continue
                if utilities[left] < utilities[right]:
                    left, right = right, left
            gap = float(utilities[left] - utilities[right])
            good.append(left)
            bad.append(right)
            left_hit = rows[left].get("rho") is not None and float(rows[left]["rho"]) <= 1.0
            right_hit = rows[right].get("rho") is not None and float(rows[right]["rho"]) <= 1.0
            crossing = left_hit != right_hit
            weights.append(min(4.0, 1.0 + gap) * (3.0 if crossing else 1.0))
    if not good:
        raise Geometry19RerankerError("No informative training pairs were constructed")
    return (
        np.asarray(good, dtype=np.int64),
        np.asarray(bad, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def _load_rows(
    ledger: Path,
    target_suite: Path,
    split_map: Mapping[str, Sequence[str]],
    maximum_rows_per_split: int | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    targets: dict[str, dict[str, Any]] = {}
    with target_suite.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            target = dict(row["target"])
            targets[str(target["target_id"])] = target
    wanted = {value for values in split_map.values() for value in values}
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    with ledger.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            split = str(row.get("split"))
            if split not in wanted or row.get("geometry19") is None:
                continue
            if maximum_rows_per_split is not None and counts.get(split, 0) >= maximum_rows_per_split:
                continue
            target_id = str(row["target_id"])
            if target_id not in targets:
                raise Geometry19RerankerError(f"Missing target definition: {target_id}")
            counts[split] = counts.get(split, 0) + 1
            rows.append(row)
    if not rows:
        raise Geometry19RerankerError("Candidate ledger has no mapped Geometry19 rows")
    return rows, targets


def _ensemble_prediction(
    model_root: Path,
    seeds: Sequence[int],
    feature_names: Sequence[str],
    x: np.ndarray,
    device: torch.device,
    *,
    shared48: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    for seed in seeds:
        model, normalization, manifest = _load_model(
            model_root / f"seed_{seed}", feature_names, device
        )
        predictions.append(
            _predict(
                model,
                x,
                normalization,
                device,
                str(manifest.get("output_transform", "identity")),
                shared48=shared48,
            )
        )
    array = np.stack(predictions)
    return np.mean(array, axis=0), np.std(array, axis=0)


def _feature_matrix(
    x: np.ndarray,
    targets: np.ndarray,
    s0_prediction: np.ndarray,
    performance_prediction: np.ndarray,
    performance_std: np.ndarray,
    tolerances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = np.maximum(np.abs(targets), 1.0e-15) * tolerances[None, :]
    s0_ratio = np.clip((s0_prediction - targets) / denominator, -100.0, 100.0)
    performance_ratio = np.clip((performance_prediction - targets) / denominator, -100.0, 100.0)
    disagreement = np.clip((performance_prediction - s0_prediction) / denominator, -100.0, 100.0)
    uncertainty = np.clip(performance_std / denominator, 0.0, 100.0)
    s0_rho = np.max(np.abs(s0_ratio), axis=1)
    performance_rho = np.max(np.abs(performance_ratio), axis=1)
    target_log = np.log(np.maximum(targets, 1.0e-15))
    features = np.concatenate(
        (
            x,
            target_log,
            s0_ratio,
            performance_ratio,
            disagreement,
            uncertainty,
            np.log1p(s0_rho)[:, None],
            np.log1p(performance_rho)[:, None],
        ),
        axis=1,
    ).astype(np.float32)
    return features, s0_rho, performance_rho


def _scores(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            values.append(model(torch.from_numpy(features[start : start + 4096]).to(device)).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def _selection_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    rank = metrics["median_first_hit_rank"]
    return (
        float(metrics["hit_bearing_recall"]["32"]),
        float(metrics["hit_bearing_recall"]["64"]),
        -float(rank if rank is not None else math.inf),
    )


def _train_one(
    seed: int,
    train_rows: Sequence[Mapping[str, Any]],
    train_features: np.ndarray,
    validation_rows: Sequence[Mapping[str, Any]],
    validation_features: np.ndarray,
    budgets: Sequence[int],
    training: Mapping[str, Any],
    device: torch.device,
) -> tuple[PairwiseReranker, dict[str, Any]]:
    _seed_all(seed)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1.0e-6] = 1.0
    train_x = ((train_features - mean) / std).astype(np.float32)
    validation_x = ((validation_features - mean) / std).astype(np.float32)
    utilities = np.empty(len(train_rows), dtype=np.float32)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(train_rows):
        grouped.setdefault(str(row["run_id"]), []).append(index)
    for indices in grouped.values():
        finite = [float(train_rows[index]["rho"]) for index in indices if train_rows[index].get("rho") is not None]
        worst = max(finite) if finite else 1000.0
        for index in indices:
            utilities[index] = true_utility(train_rows[index].get("rho"), worst)
    model = PairwiseReranker(train_x.shape[1], training["hidden_layers"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        good, bad, weights = build_pair_indices(
            train_rows,
            utilities,
            seed=seed * 1000 + epoch,
            pairs_per_run=int(training["pairs_per_run"]),
            elite_per_run=int(training["elite_per_run"]),
            minimum_gap=float(training["minimum_utility_gap"]),
        )
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(good))
        model.train()
        losses = []
        batch_size = int(training["batch_size"])
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            left = torch.from_numpy(train_x[good[chosen]]).to(device)
            right = torch.from_numpy(train_x[bad[chosen]]).to(device)
            weight = torch.from_numpy(weights[chosen]).to(device)
            loss = (F.softplus(-(model(left) - model(right))) * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = ranking_metrics(validation_rows, _scores(model, validation_x, device), budgets)
        history.append({"epoch": epoch, "pairwise_loss": float(np.mean(losses)), "validation": metrics})
        if best_metrics is None or _selection_key(metrics) > _selection_key(best_metrics):
            best_metrics = metrics
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(training["patience"]):
                break
    if best_state is None or best_metrics is None:
        raise Geometry19RerankerError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.feature_mean = mean
    model.feature_std = std
    return model, {"seed": seed, "best_epoch": best_epoch, "validation": best_metrics, "history": history}


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# WP19-23 独立候选排序模型实验",
        "",
        f"> 状态：**{summary['decision']['status']}**  ",
        "> 按目标隔离 train/development/locked-test；locked-test 不参与 checkpoint 或 seed 选择。",
        "",
        "## 同候选池、等预算结果",
        "",
        "| Split | Scorer | Recall@32 | Recall@64 | Recall@128 | Recall@256 | 首命中排名P50 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "locked_test", "external_regression"):
        for scorer in ("shared48_s0", "performance3_log", "independent_reranker"):
            metrics = summary["evaluation"][split][scorer]
            recall = metrics["hit_bearing_recall"]
            rank = metrics["median_first_hit_rank"]
            lines.append(
                f"| {split} | {scorer} | {recall['32']:.1%} | {recall['64']:.1%} | "
                f"{recall['128']:.1%} | {recall['256']:.1%} | {'N/A' if rank is None else format(rank, '.1f')} |"
            )
    gain = summary["decision"]["locked_recall_absolute_gain"]
    lines += [
        "",
        "## 判定",
        "",
        f"- validation 选择 seed：{summary['selected_seed']}；",
        f"- locked-test Recall@32/64 绝对变化：{gain['32']:+.1%}/{gain['64']:+.1%}；",
        f"- 相对 S0 的预注册门槛：{'达到候选门槛' if summary['decision']['promote'] else '未达到'}；pipeline 默认排序未改变。",
        "",
    ]
    return "\n".join(lines)


def run_reranker_experiment(
    root: str | Path,
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    device: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise Geometry19RerankerError("Protocol mismatch")
    runtime = Path(output_root).resolve() if output_root else repo / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise Geometry19RerankerError(f"Refusing to overwrite {runtime}")
    runtime.mkdir(parents=True)
    maximum = int(config["smoke"]["maximum_rows_per_split"]) if smoke else None
    candidate_root = repo / config["candidate_runtime"]
    rows, targets_by_id = _load_rows(
        candidate_root / config["candidate_ledger"],
        candidate_root / config["target_suite"],
        config["target_splits"],
        maximum,
    )
    feature_names = list(Geometry19.coordinate_names())
    x = np.asarray(
        [[row["geometry19"][name] for name in feature_names] for row in rows], dtype=np.float32
    )
    targets = np.asarray(
        [[targets_by_id[str(row["target_id"])][name] for name in PERFORMANCE_NAMES] for row in rows],
        dtype=np.float64,
    )
    inverse = yaml.safe_load((repo / config["inverse_protocol"]).read_text(encoding="utf-8"))
    tolerances = np.asarray(
        [inverse["diagnostic_acceptance"]["relative_tolerances"][name] for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    torch_device = torch.device(device or config["training"]["device"])
    forward_seeds = [int(value) for value in config["forward_model_seeds"]]
    s0_prediction, _ = _ensemble_prediction(
        repo / config["baseline_model_root"], forward_seeds, feature_names, x, torch_device, shared48=True
    )
    performance_prediction, performance_std = _ensemble_prediction(
        repo / config["performance_model_root"], forward_seeds, feature_names, x, torch_device, shared48=False
    )
    raw_features, s0_rho, performance_rho = _feature_matrix(
        x, targets, s0_prediction, performance_prediction, performance_std, tolerances
    )
    split_indices: dict[str, np.ndarray] = {}
    for name, source_splits in config["target_splits"].items():
        allowed = set(source_splits)
        split_indices[name] = np.asarray(
            [index for index, row in enumerate(rows) if row["split"] in allowed], dtype=np.int64
        )
    target_sets = {
        name: {str(rows[index]["target_id"]) for index in indices}
        for name, indices in split_indices.items()
    }
    names = list(target_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = target_sets[left] & target_sets[right]
            if overlap:
                raise Geometry19RerankerError(f"Target leakage between {left} and {right}")
    training = dict(config["training"])
    seeds = [int(value) for value in config["reranker_seeds"]]
    if smoke:
        seeds = [int(value) for value in config["smoke"]["reranker_seeds"]]
        training.update(config["smoke"]["training"])
    train_index = split_indices["train"]
    validation_index = split_indices["validation"]
    trained: list[tuple[PairwiseReranker, dict[str, Any]]] = []
    for seed in seeds:
        trained.append(
            _train_one(
                seed,
                [rows[index] for index in train_index],
                raw_features[train_index],
                [rows[index] for index in validation_index],
                raw_features[validation_index],
                [int(value) for value in config["ranking_budgets"]],
                training,
                torch_device,
            )
        )
    selected_model, selected_run = max(trained, key=lambda item: _selection_key(item[1]["validation"]))
    selected_seed = int(selected_run["seed"])
    model_dir = runtime / "models" / f"seed_{selected_seed}"
    model_dir.mkdir(parents=True)
    torch.save(selected_model.state_dict(), model_dir / "reranker_state.pt")
    np.savez(
        model_dir / "feature_normalization.npz",
        mean=selected_model.feature_mean,
        std=selected_model.feature_std,
    )
    budgets = [int(value) for value in config["ranking_budgets"]]
    evaluation: dict[str, Any] = {}
    for split_name in ("validation", "locked_test", "external_regression"):
        indices = split_indices[split_name]
        split_rows = [rows[index] for index in indices]
        normalized = (
            (raw_features[indices] - selected_model.feature_mean) / selected_model.feature_std
        ).astype(np.float32)
        evaluation[split_name] = {
            "shared48_s0": ranking_metrics(split_rows, -s0_rho[indices], budgets),
            "performance3_log": ranking_metrics(split_rows, -performance_rho[indices], budgets),
            "independent_reranker": ranking_metrics(
                split_rows, _scores(selected_model, normalized, torch_device), budgets
            ),
        }
    baseline_locked = evaluation["locked_test"]["shared48_s0"]["hit_bearing_recall"]
    reranker_locked = evaluation["locked_test"]["independent_reranker"]["hit_bearing_recall"]
    baseline_validation = evaluation["validation"]["shared48_s0"]["hit_bearing_recall"]
    reranker_validation = evaluation["validation"]["independent_reranker"]["hit_bearing_recall"]
    validation_gain = float(reranker_validation["32"] - baseline_validation["32"])
    locked_gain = {str(k): float(reranker_locked[str(k)] - baseline_locked[str(k)]) for k in budgets}
    policy = config["promotion"]
    promote = bool(
        not smoke
        and validation_gain >= float(policy["minimum_validation_recall_at_32_absolute_gain"])
        and locked_gain["32"] >= float(policy["minimum_locked_recall_at_32_absolute_gain"])
        and (
            not policy["require_locked_recall_at_64_non_degradation"]
            or locked_gain["64"] >= 0.0
        )
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "SMOKE_COMPLETE" if smoke else "COMPLETE",
        "formal_use_allowed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "feature_count": int(raw_features.shape[1]),
        "split_row_counts": {name: len(indices) for name, indices in split_indices.items()},
        "split_target_counts": {name: len(values) for name, values in target_sets.items()},
        "selected_seed": selected_seed,
        "training_runs": [run for _, run in trained],
        "evaluation": evaluation,
        "decision": {
            "status": "PROMOTION_CANDIDATE" if promote else "NO_PROMOTION",
            "promote": promote,
            "validation_recall_at_32_absolute_gain": validation_gain,
            "locked_recall_absolute_gain": locked_gain,
            "default_pipeline_changed": False,
        },
    }
    _write_json(runtime / config["artifacts"]["summary"], summary)
    _write_text(runtime / config["artifacts"]["report"], _render(summary))
    return summary
