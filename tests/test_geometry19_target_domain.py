"""Evidence-backed simplified-V2.2 target-domain tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_target_domain import (
    AXES,
    DOMAIN_VERSION,
    Geometry19TargetDomainError,
    load_geometry19_target_domain,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_PATH = (
    REPO_ROOT
    / "config/physics/geometry19_v2_2_development_target_domain_v1.yaml"
)
INTERPOLATION_PATH = (
    REPO_ROOT
    / "config/experiments/geometry19_v2_2_target_domain_interpolation_v1.yaml"
)


def test_target_domain_loads_evidence_and_accepts_every_vertex() -> None:
    domain = load_geometry19_target_domain(REPO_ROOT)

    assert domain.source_hash
    assert len(domain.nodes) == 6
    assert len(domain.segments) == 5
    for values in domain.nodes.values():
        assessment = domain.assess(dict(zip(AXES, values)))
        assert assessment.inside is True
        assert assessment.normalized_distance == pytest.approx(0.0, abs=1.0e-10)


def test_target_domain_accepts_all_validated_interpolation_points() -> None:
    domain = load_geometry19_target_domain(REPO_ROOT)
    campaign = yaml.safe_load(INTERPOLATION_PATH.read_text(encoding="utf-8"))

    assessments = [domain.assess(target) for target in campaign["targets"]]

    assert all(value.inside for value in assessments)
    assert all(value.normalized_distance <= 1.0 for value in assessments)
    assert {value.nearest_segment for value in assessments} == {
        "low_mass_low_to_mid",
        "high_mass_low_to_mid",
        "cross_branch_mid",
        "low_mass_mid_to_high",
        "high_mass_mid_to_high",
    }


@pytest.mark.parametrize(
    "target",
    [
        {
            "thrust_N": 86.31618713298968,
            "tsfc_kg_N_s": 0.000056389888441994895,
            "mass_kg": 2.5609349639240773,
        },
        {
            "thrust_N": 109.48969371415566,
            "tsfc_kg_N_s": 0.0000475211450638959,
            "mass_kg": 2.8746889370990707,
        },
        {
            "thrust_N": 169.23418951967648,
            "tsfc_kg_N_s": 0.000035284841418524566,
            "mass_kg": 1.7129864897096319,
        },
    ],
)
def test_target_domain_rejects_observed_unstable_targets(target: dict) -> None:
    assessment = load_geometry19_target_domain(REPO_ROOT).assess(target)

    assert assessment.inside is False
    assert assessment.status == "OUTSIDE_V2_2_DEVELOPMENT_TARGET_DOMAIN"
    assert assessment.normalized_distance > 1.0


def test_target_domain_rejects_evidence_hash_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(DOMAIN_PATH.read_text(encoding="utf-8"))
    payload["evidence"]["vertex_campaign"]["sha256"] = "0" * 64
    source = tmp_path / "domain.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(Geometry19TargetDomainError, match="evidence hash mismatch"):
        load_geometry19_target_domain(REPO_ROOT, source)


def test_target_domain_contract_remains_development_only() -> None:
    payload = yaml.safe_load(DOMAIN_PATH.read_text(encoding="utf-8"))

    assert payload["domain_version"] == DOMAIN_VERSION
    assert payload["formal_use_allowed"] is False
    assert payload["engineering_validated"] is False
    assert payload["interpolation_only"] is True
    assert payload["extrapolation_allowed"] is False
