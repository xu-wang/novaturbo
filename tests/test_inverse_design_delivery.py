from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from src.ai.diag_cfm_strategy import load_diag_cfm_strategy_config
from src.ai.inverse_design_delivery import (
    compose_inverse_design_config as compose_high_precision_config,
)
from src.ai.inverse_design_delivery_cfm4_legacy import (
    compose_inverse_design_config,
    run_inverse_design_delivery,
)
from src.ai.stage2_wp5 import _load_target, load_stage2_wp5_config, run_stage2_wp5


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "config" / "inverse_design" / "legacy" / "cfm4_default_v1.yaml"
)
HIGH_PRECISION_CONFIG = REPO_ROOT / "config" / "inverse_design" / "default.yaml"
DEFAULT_TARGET = (
    REPO_ROOT
    / "config"
    / "inverse_design"
    / "targets"
    / "examples"
    / "default_ordinary.yaml"
)


def _write_yaml(path: Path, value) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_public_default_is_complete_high_precision_strategy(tmp_path):
    composed, sources = compose_high_precision_config(
        HIGH_PRECISION_CONFIG,
        DEFAULT_TARGET,
        repo_root=REPO_ROOT,
    )
    assert composed["strategy_id"] == "cfm4_diag_prefilter_gradient_nsga_v1"
    assert composed["generation"]["diag_prefilter"]["enabled"] is True
    assert composed["generation"]["diag_prefilter"]["pool_size"] == 512
    assert composed["gradient"]["enabled"] is True
    assert composed["nsga2"]["enabled"] is True
    assert set(sources) == {"delivery", "target"}
    path = tmp_path / "composed.yaml"
    _write_yaml(path, composed)
    load_diag_cfm_strategy_config(path)


def test_public_default_postprocessing_can_be_disabled_independently(tmp_path):
    composed, _ = compose_high_precision_config(
        HIGH_PRECISION_CONFIG,
        DEFAULT_TARGET,
        repo_root=REPO_ROOT,
        gradient_enabled_override=False,
        nsga2_enabled_override=False,
    )
    assert composed["gradient"]["enabled"] is False
    assert composed["nsga2"]["enabled"] is False
    path = tmp_path / "composed_disabled.yaml"
    _write_yaml(path, composed)
    loaded = load_diag_cfm_strategy_config(path)
    assert loaded["gradient"]["enabled"] is False
    assert loaded["nsga2"]["enabled"] is False


def test_legacy_components_reproduce_frozen_numerical_configuration():
    composed, sources = compose_inverse_design_config(
        DEFAULT_CONFIG,
        DEFAULT_TARGET,
        repo_root=REPO_ROOT,
    )
    legacy = load_stage2_wp5_config(
        REPO_ROOT / "config" / "stage2" / "s2_wp5_cfm_default.yaml"
    )
    for key in (
        "data",
        "model",
        "generation",
        "delivery",
        "gradient",
        "latent_repair",
        "nsga2",
        "runtime",
    ):
        assert composed[key] == legacy[key]
    assert _load_target(REPO_ROOT, composed["target"]) == _load_target(
        REPO_ROOT, legacy["target"]
    )
    assert set(sources) == {"delivery", "policy", "generator", "target"}


def test_extreme_target_and_seed_reproduce_legacy_branch_configuration(tmp_path):
    delivery = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    delivery["generation"]["seed"] = 20913717
    delivery_path = tmp_path / "extreme_delivery.yaml"
    _write_yaml(delivery_path, delivery)
    target_path = (
        REPO_ROOT
        / "config"
        / "inverse_design"
        / "targets"
        / "frozen"
        / "cfm_default_extreme.yaml"
    )
    composed, _ = compose_inverse_design_config(
        delivery_path,
        target_path,
        repo_root=REPO_ROOT,
    )
    legacy = load_stage2_wp5_config(
        REPO_ROOT / "config" / "experiments" / "bounded_flows" / "extreme_cfm_default.yaml"
    )
    for key in (
        "data",
        "model",
        "generation",
        "delivery",
        "gradient",
        "latent_repair",
        "nsga2",
        "runtime",
    ):
        assert composed[key] == legacy[key]
    assert _load_target(REPO_ROOT, composed["target"]) == _load_target(
        REPO_ROOT, legacy["target"]
    )


def test_legacy_facade_and_frozen_entry_produce_identical_direct_delivery(
    tmp_path, monkeypatch
):
    legacy = copy.deepcopy(
        load_stage2_wp5_config(
            REPO_ROOT / "config" / "stage2" / "s2_wp5_cfm_default.yaml"
        )
    )
    legacy["runtime"]["device"] = "cpu"
    legacy_path = tmp_path / "legacy.yaml"
    _write_yaml(legacy_path, legacy)

    delivery = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    delivery["runtime"]["device"] = "cpu"
    delivery_path = tmp_path / "delivery.yaml"
    _write_yaml(delivery_path, delivery)

    monkeypatch.setattr("src.ai.stage2_wp5.collect_environment", lambda: {})
    monkeypatch.setattr("src.ai.stage2_wp5.collect_git_state", lambda *_args: {})
    monkeypatch.setattr("src.ai.stage2_wp5.collect_source_manifest", lambda *_args: [])

    legacy_output = run_stage2_wp5(
        legacy_path,
        repo_root=REPO_ROOT,
        output_dir_override=tmp_path / "legacy_output",
    )
    stable_output = run_inverse_design_delivery(
        delivery_path,
        DEFAULT_TARGET,
        repo_root=REPO_ROOT,
        output_dir_override=tmp_path / "stable_output",
    )

    identical_artifacts = (
        "generation/raw_candidates.csv",
        "generation/validation.csv",
        "direct/forward_results.csv",
        "direct/ranked_candidates.csv",
        "delivery/delivered_candidates.csv",
        "delivery/evaluation_cost_ledger.csv",
        "delivery/decision.json",
    )
    for relative in identical_artifacts:
        assert (stable_output / relative).read_bytes() == (legacy_output / relative).read_bytes()

    decision = json.loads(
        (stable_output / "delivery" / "decision.json").read_text(encoding="utf-8")
    )
    assert decision["executed_actions"] == []
    assert decision["cost"]["total_0d"] == 64
    provenance = yaml.safe_load(
        (stable_output / "configuration" / "sources.yaml").read_text(encoding="utf-8")
    )
    assert set(provenance["sources"]) == {"delivery", "policy", "generator", "target"}
