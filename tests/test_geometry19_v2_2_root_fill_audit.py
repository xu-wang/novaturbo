"""Evidence checks for the cumulative V2.2 development root-fill audit."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "data/validation/v2_2_root_fill_audit_v1"


def test_frozen_root_fill_audit_records_partial_bridge_coverage() -> None:
    summary_path = EVIDENCE / "audit_summary.json"
    report_path = EVIDENCE / "Root_Fill_Audit.md"
    audit = json.loads(summary_path.read_text(encoding="utf-8"))
    assert audit["status"] == "DEVELOPMENT_ROOT_FILL_PASS_BRIDGE_COVERAGE_PARTIAL"
    assert audit["root_count_gate_pass"] is True
    assert audit["fold_gate_pass"] is True
    assert audit["bridge_coverage_gate_pass"] is False
    assert audit["confirmation_excluded"] is True
    assert audit["confirmation_reuse_allowed"] is False
    assert audit["development_ablation_ready"] is True
    assert audit["counts"]["final_development_rows"] == 3489
    assert audit["counts"]["final_global_roots"] == 52
    for segment in ("low_mass_low_to_mid", "low_mass_mid_to_high"):
        assert audit["segments"][segment]["original_roots"] == 12
        assert audit["segments"][segment]["final_roots"] == 27
        assert audit["segments"][segment]["added_roots"] == 15
    assert (
        audit["segments"]["low_mass_low_to_mid"]["cumulative_bridge_coverage"]
        ["p95_reduction_fraction"]
        > 0.20
    )
    assert (
        audit["segments"]["low_mass_mid_to_high"]["cumulative_bridge_coverage"]
        ["p95_reduction_fraction"]
        < 0.02
    )
    assert sha256(report_path.read_bytes()).hexdigest()
