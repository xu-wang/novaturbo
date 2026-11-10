from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_phase_a_audit_covers_required_paths_and_boundaries():
    audit_path = REPO_ROOT / "docs" / "NovaTurbo_Current_Implementation_Audit.md"
    audit = audit_path.read_text(encoding="utf-8")

    required_paths = (
        "config/stage2/s2_wp5_cfm_default.yaml",
        "config/stage2/default_inverse_design_policy_v1.yaml",
        "scripts/run_stage2_wp5_cfm.py",
        "src/ai/stage2_wp5.py",
        "src/ai/inverse_design.py",
        "src/ai/generators/",
        "src/ai/surrogate.py",
        "src/physics/brayton.py",
        "src/cfd/",
    )
    assert all(path in audit for path in required_paths)
    assert "Phase A 完成" in audit
    assert "未实现 D-Flow" in audit
    assert "没有 B-Rep/OCCT STEP exporter" in audit
    assert "仓库不内置 CFD solver" in audit


def test_canonical_and_legacy_inverse_design_roles_are_explicit():
    legacy_policy_path = (
        REPO_ROOT / "config" / "stage2" / "default_inverse_design_policy_v1.yaml"
    )
    legacy_policy = yaml.safe_load(legacy_policy_path.read_text(encoding="utf-8"))

    assert legacy_policy["status"] == "superseded_legacy_reference"
    legacy = legacy_policy["legacy_execution"]
    assert legacy["script"] == "scripts/run_inverse_design_cfm4_legacy.py"
    assert legacy["config"] == "config/inverse_design/legacy/cfm4_default_v1.yaml"
    assert legacy["target_argument"] == "--target-config"
    assert legacy_policy["superseded_by"]["script"] == "scripts/run_inverse_design.py"
