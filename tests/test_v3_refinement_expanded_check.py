import json
from pathlib import Path

import pytest

from src.ai import v3_refinement_expanded_check as x

ROOT = Path(__file__).resolve().parents[1]


def test_expanded_queue_covers_all_centers_without_repeating_previous_targets():
    _, _, _, support, norm, _ = x.p.source(ROOT)
    rows = [json.loads(s) for s in (ROOT/x.REGISTRY).read_text().splitlines()]
    excluded = {r['target_id'] for r in x.p.read(ROOT/x.PREVIOUS_TARGETS)}
    selected = x.choose_targets(rows, support, norm, excluded)
    assert len(selected) == 98
    assert len({r['source_center_id'] for r in selected}) == 55
    assert not ({r['target_id'] for r in selected} & excluded)
    assert all(r['internal_split']=='tuning' and 0<=r['source_support_distance_ratio']<=1 for r in selected)
    assert selected == x.choose_targets(list(reversed(rows)), list(reversed(support)), norm, excluded)


def test_expanded_protocol_lock_rejects_changed_queue_and_dependency(tmp_path, monkeypatch):
    output=tmp_path/'expanded'
    x.freeze(ROOT,output)
    x.check_lock(ROOT,output)
    monkeypatch.setattr(x,'dependencies',lambda _: {})
    with pytest.raises(ValueError, match='dependencies changed'):
        x.check_lock(ROOT,output)
    x.p.write(output/'targets.json',[])
    with pytest.raises(ValueError,match='target queue changed'):
        x.check_lock(ROOT,output)


@pytest.mark.parametrize('c_hit',[True,False])
def test_matched_control_reuses_c_and_evaluates_only_next_eight(c_hit,monkeypatch):
    def record(i):
        return dict(design_id=str(i),rho=.1 if c_hit else 2., diagnostic_domain_valid=True,
                    software_hit_5pct=c_hit,feasible=False)
    c=[record(i) for i in range(8)]
    b=[record(i) for i in range(24)]
    target={'target_id':'public_only'}
    contents={'C/evaluations.json':c,'B/candidates.json':b,'target.json':target}
    monkeypatch.setattr(x.p,'read',lambda path:contents[str(path)])
    calls=[]
    class Profile:
        @staticmethod
        def from_yaml(_):return object()
    monkeypatch.setattr(x.p,'ConstraintProfile',Profile)
    monkeypatch.setattr(x.p,'build_v3_development_physics_runtime',lambda _:object())
    def evaluate(row,*_):
        calls.append(row['design_id'])
        return {**row,'rho':.1,'software_hit_5pct':True}
    monkeypatch.setattr(x.p,'evaluate_design',evaluate)
    result=x.control_run(Path('.'),Path('.'),dict(sources={'diagnostic_profile_path':'profile'},f_recompute=x.p.F_SELECTION_POLICY))
    if c_hit:
        assert result=={'evaluations':[],'independent_recompute':None} and calls==[]
    else:
        assert calls[:8]==[str(i) for i in range(8,16)] and len(calls)==9
        assert len(result['evaluations'])==8
        assert result['independent_recompute']['design_id']=='10'  # Equal rho uses lexicographic IDs.
