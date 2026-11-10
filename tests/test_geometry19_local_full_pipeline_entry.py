import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from src.ai import geometry19_local_full_pipeline_entry as m


def test_stage_contract_detects_tamper(tmp_path):
    seal = m.seal_stage(tmp_path / "C", {"stage": "C", "status": "HIT"}, [{"x": 1}])
    assert seal["stage"] == "C"
    m.verify_stage(tmp_path / "C", "C")
    (tmp_path / "C" / "ledger.jsonl").write_text("{}\n")
    with pytest.raises(Exception, match="artifact changed"):
        m.verify_stage(tmp_path / "C", "C")


def test_target_resume_requires_complete_manifest(tmp_path):
    target = tmp_path / "target000"
    target.mkdir()
    with pytest.raises(Exception):
        m.verify_target(target, "lock")


def test_fixed_stage_order():
    assert m.STAGES == ("B", "C", "D", "E", "F")


def test_delivery_uses_exact_serialized_geometry(tmp_path, monkeypatch):
    geometry = {"x": 1.0000000000000002}
    task = SimpleNamespace(
        output=tmp_path, dec={},
        context=SimpleNamespace(physics_runtime=SimpleNamespace(model_config=object())),
    )
    monkeypatch.setattr(m.g3.adapter, "_assign_pooled_support", lambda value, dec, radius: ({"root_seed_id": "r", "root_index": 0}, .2))
    monkeypatch.setattr(m, "Geometry19", SimpleNamespace(from_mapping=lambda value: SimpleNamespace(as_mapping=lambda: value)))
    monkeypatch.setattr(m, "derive_geometry_state", lambda design, config: object())
    task.evaluate = lambda row, stage, force: {
        "development_diagnostic_hit": True, "performance": {"thrust_N": 1., "tsfc_kg_N_s": 1., "mass_kg": 1.},
    }
    reference = {"geometry19": geometry, "design_hash": "d", "performance": {"thrust_N": 1., "tsfc_kg_N_s": 1., "mass_kg": 1.}}
    result = m.deliver_exact_geometry(task, reference)
    assert result["passed"]
    assert json.loads((tmp_path / "Delivery_Geometry.json").read_text()) == geometry
