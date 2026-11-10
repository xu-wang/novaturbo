from pathlib import Path

import pandas as pd

from src.ai.stage2_wp6 import (
    _bootstrap_ci,
    _delivery_matrix,
    _markdown_table,
    _quantile_failure_aware,
    _paired_end_to_end,
    load_stage2_wp6_config,
    prepare_stage2_wp6,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "stage2" / "s2_wp6_g3_v2.yaml"


def test_wp6_config_and_formal_delivery_matrix():
    config = load_stage2_wp6_config(CONFIG_PATH)
    suite = pd.read_csv(
        REPO_ROOT / "data/generator_active10d_100k/targets/target_suite_active10d_v1.csv"
    )
    frozen = __import__("yaml").safe_load(
        (REPO_ROOT / "config/stage2/s2_wp1_end_to_end_targets_v1.yaml").read_text()
    )
    targets = suite.loc[suite["target_id"].astype(str).isin(frozen["target_ids"])]
    matrix = _delivery_matrix(config, targets)
    assert len(matrix) == 320
    assert set(matrix["model"]) == {"nflows", "cfm"}
    assert set(matrix["training_seed"]) == {42, 43}
    assert set(matrix["optimizer_seed"]) == {42, 43}
    assert set(matrix["sample_seed"]) == {20260816, 20260817}
    assert not matrix["run_id"].duplicated().any()


def test_wp6_smoke_preparation_is_small_and_auditable(tmp_path):
    output = prepare_stage2_wp6(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        output_dir_override=tmp_path / "wp6_smoke",
        maximum_targets_per_type=1,
    )
    manifest = __import__("yaml").safe_load((output / "manifest.yaml").read_text())
    assert manifest["smoke"] is True
    assert manifest["generator_only_target_count"] == 4
    assert manifest["end_to_end_target_count"] == 2
    assert manifest["planned_generator_only_runs"] == 16
    assert manifest["planned_benchmark_runs"] == 24
    assert manifest["planned_delivery_runs"] == 16


def test_bootstrap_ci_is_deterministic_and_ordered():
    first = _bootstrap_ci(
        [-2.0, -1.0, 0.0, 1.0], statistic="median", replicates=500,
        seed=42, confidence=0.95,
    )
    second = _bootstrap_ci(
        [-2.0, -1.0, 0.0, 1.0], statistic="median", replicates=500,
        seed=42, confidence=0.95,
    )
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_markdown_table_needs_no_optional_dependency():
    rendered = _markdown_table(pd.DataFrame([{"method": "cfm", "score": 0.125}]))
    assert "| method | score |" in rendered
    assert "| cfm | 0.125 |" in rendered


def test_failure_aware_quantile_retains_infinite_failures_without_interpolation():
    values = pd.Series([0.1] * 94 + [float("inf")] * 6)
    assert _quantile_failure_aware(values, 0.5) == 0.1
    assert _quantile_failure_aware(values, 0.95) == float("inf")


def test_end_to_end_pairing_supports_delivery_and_search_baselines():
    config = load_stage2_wp6_config(CONFIG_PATH)
    keys = {
        "training_seed": 42,
        "target_id": "target",
        "target_type": "ID",
        "optimizer_seed": 42,
    }
    delivery = pd.DataFrame([
        {
            **keys, "model": "cfm", "final_best_error": 0.01,
            "failure_aware_first_hit": 10.0, "total_0d": 64,
        },
        {
            **keys, "model": "nflows", "final_best_error": 0.02,
            "failure_aware_first_hit": 12.0, "total_0d": 32,
        },
    ])
    benchmark = pd.DataFrame([
        {
            **keys, "method": "screened_random", "search_best_error": 0.03,
            "search_evaluations_to_hit": 100.0, "prefilter_evaluations": 128,
            "total_evaluations": 768,
        },
        {
            **keys, "method": "random", "search_best_error": 0.04,
            "search_evaluations_to_hit": 200.0, "prefilter_evaluations": 0,
            "total_evaluations": 640,
        },
    ])
    paired = _paired_end_to_end(delivery, benchmark, config)
    assert set(paired["baseline"]) == {"nflows", "screened_random", "random"}
    assert paired["primary_error_win"].all()
    assert set(paired["primary_total_0d"]) == {64}
