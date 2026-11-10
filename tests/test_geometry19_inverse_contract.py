"""WP19-23 frozen target, acceptance, and artifact contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_inverse_contract import (
    DIAGNOSTIC_PROFILE,
    Geometry19InverseContractError,
    Geometry19InverseTarget,
    TARGET_CONTRACT_VERSION,
    derive_target_id,
    load_wp19_23_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "config" / "physics" / "wp19_23_inverse_closure_v1.yaml"


def _target_payload() -> dict:
    payload = {
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "target_id": "",
        "target_type": "ID",
        "expected_status": "KNOWN_FEASIBLE",
        "thrust_N": 100.0,
        "tsfc_kg_N_s": 4.0e-5,
        "mass_kg": 1.5,
        "acceptance_profile": DIAGNOSTIC_PROFILE,
    }
    payload["target_id"] = derive_target_id(payload)
    return payload


def test_frozen_protocol_preserves_fixed_raw_and_development_only_semantics() -> None:
    contract = load_wp19_23_contract(PROTOCOL)
    payload = contract.payload

    assert payload["generation"]["raw_candidate_count"] == 256
    assert payload["generation"]["generator_call_count_per_target"] == 1
    assert payload["generation"]["refill_allowed"] is False
    assert payload["generation"]["candidate_copy_allowed"] is False
    assert payload["candidate_filter"]["trim_rule"] == (
        "largest_power_of_two_not_above_valid_count"
    )
    assert payload["candidate_filter"]["discrete_policy"] == (
        "unconditional_nearest_integer"
    )
    assert payload["candidate_filter"]["discrete_absolute_tolerance_count"] is None
    assert payload["candidate_filter"]["surrogate_reranking_before_trim_allowed"] is True
    assert payload["candidate_filter"]["select_before_physics"] == (
        "predicted_rho_ascending_then_candidate_id"
    )
    assert payload["hit_alignment"]["hard_metric"] == "rho_max_axis_ratio"
    assert payload["hit_alignment"]["true_acceptance"] == (
        "lexicographic_failed_axes_rho_sum_constraint"
    )
    assert payload["optimizer"]["selection_order"] == (
        "failed_axes_then_rho_then_sum_then_constraint_violation"
    )
    generator = payload["models"]["generator"]
    assert generator["physical_gpu"] == 6
    assert generator["default_variant"] == "bounded-sigmoid"
    assert list(generator["variants"]) == ["baseline", "bounded-sigmoid"]
    assert contract.default_generator_variant == "bounded-sigmoid"
    assert contract.generator_variant()["model_tag"].endswith("bounded_sigmoid_v1")
    assert contract.generator_variant("baseline")["training_seed"] == 43
    forward = payload["models"]["forward"]
    assert forward["default_tag"] == "performance3_log"
    assert list(forward["variants"]) == ["performance3_log", "s0"]
    assert contract.default_forward_model_tag == "performance3_log"
    assert contract.forward_model()["output_contract"] == "performance3_only"
    assert contract.forward_model("s0")["use_constraint_loss"] is True
    assert payload["generation"]["supported_sampling_modes"] == ["direct", "dflow"]
    assert "generator_normalizer_hash" in payload["artifact_protocol"][
        "manifest_required_hashes"
    ]
    assert "feasibility_checkpoint_hash" not in payload["artifact_protocol"][
        "manifest_required_hashes"
    ]
    assert payload["artifact_protocol"]["manifest_conditional_hashes"] == {
        "feasibility_checkpoint_hash": (
            "required_when_forward_variant_uses_feasibility_loss"
        )
    }
    assert contract.diagnostic_acceptance.formal_hit_rate_ready is False


def test_s0_can_be_selected_as_config_default(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["models"]["forward"]["default_tag"] = "s0"
    alternate = tmp_path / "wp19_23_s0_default.yaml"
    alternate.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    contract = load_wp19_23_contract(alternate)
    assert contract.default_forward_model_tag == "s0"
    assert contract.forward_model()["output_contract"] == (
        "performance3_plus_constraint_residuals"
    )


def test_target_id_and_exact_field_contract_are_strict() -> None:
    payload = _target_payload()
    target = Geometry19InverseTarget.from_mapping(payload)
    assert target.target_id == derive_target_id(payload)

    changed = dict(payload, mass_kg=1.6)
    with pytest.raises(Geometry19InverseContractError, match="Target ID"):
        Geometry19InverseTarget.from_mapping(changed)

    extra = dict(payload, notes="not part of the canonical target")
    with pytest.raises(Geometry19InverseContractError, match="fields mismatch"):
        Geometry19InverseTarget.from_mapping(extra)


def test_diagnostic_hit_requires_all_three_axes_and_true_physics_gates() -> None:
    contract = load_wp19_23_contract(PROTOCOL)
    target = Geometry19InverseTarget.from_mapping(_target_payload())
    tolerances = contract.diagnostic_acceptance.relative_tolerances
    passing = {
        "thrust_N": target.thrust_N * (1.0 + 0.99 * tolerances["thrust_N"]),
        "tsfc_kg_N_s": target.tsfc_kg_N_s
        * (1.0 - 0.99 * tolerances["tsfc_kg_N_s"]),
        "mass_kg": target.mass_kg * (1.0 + 0.99 * tolerances["mass_kg"]),
    }
    assert contract.diagnostic_acceptance.is_diagnostic_hit(
        target,
        passing,
        solver_converged=True,
        numerical_hard_constraints_passed=True,
    )

    failed_axis = dict(passing, mass_kg=target.mass_kg * (1.01 + tolerances["mass_kg"]))
    assert not contract.diagnostic_acceptance.is_diagnostic_hit(
        target,
        failed_axis,
        solver_converged=True,
        numerical_hard_constraints_passed=True,
    )
    assert not contract.diagnostic_acceptance.is_diagnostic_hit(
        target,
        passing,
        solver_converged=False,
        numerical_hard_constraints_passed=True,
    )


def test_artifact_protocol_is_complete_and_refuses_overwrite() -> None:
    contract = load_wp19_23_contract(PROTOCOL)
    assert contract.payload["artifact_protocol"]["refuse_overwrite"] is True
    assert set(contract.required_run_artifacts) == {
        "manifest.json",
        "target.json",
        "raw_candidates.jsonl",
        "candidate_filter_ledger.jsonl",
        "physics_forward_ledger.jsonl",
        "candidate_disposition_ledger.jsonl",
        "refinement_ledger.jsonl",
        "nsga2_ledger.jsonl",
        "delivered_candidates.json",
        "summary.json",
        "ui_candidates",
        "pipeline_manifest.json",
    }


def test_contract_rejects_formal_hit_or_refill_mutation(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["diagnostic_acceptance"]["formal_hit_rate_ready"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19InverseContractError, match="Formal hit rate"):
        load_wp19_23_contract(invalid)

    payload["diagnostic_acceptance"]["formal_hit_rate_ready"] = False
    payload["generation"]["refill_allowed"] = True
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19InverseContractError, match="Refill"):
        load_wp19_23_contract(invalid)


def test_contract_rejects_reenabling_discrete_tolerance(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["candidate_filter"]["discrete_absolute_tolerance_count"] = 0.3
    invalid = tmp_path / "invalid_discrete_tolerance.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19InverseContractError, match="Unconditional discrete projection"):
        load_wp19_23_contract(invalid)


def test_contract_rejects_generator_default_or_variant_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["models"]["generator"]["default_variant"] = "baseline"
    invalid = tmp_path / "invalid_generator_default.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19InverseContractError, match="Bounded-sigmoid"):
        load_wp19_23_contract(invalid)

    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["models"]["generator"]["variants"]["bounded-sigmoid"][
        "checkpoint"
    ] = "wrong.pt"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19InverseContractError, match="variant metadata"):
        load_wp19_23_contract(invalid)


def test_contract_rejects_constraint_dominance_mutation(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["optimizer"]["constraint_dominance"] = "unfrozen"
    invalid = tmp_path / "invalid_constraint_dominance.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19InverseContractError, match="constraint dominance"):
        load_wp19_23_contract(invalid)


def test_contract_rejects_hit_alignment_mutation(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["hit_alignment"]["hard_metric"] = "mean_relative_error"
    invalid = tmp_path / "invalid_hit_alignment.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19InverseContractError, match="hard metric"):
        load_wp19_23_contract(invalid)
