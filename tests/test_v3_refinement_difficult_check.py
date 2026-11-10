import json
from pathlib import Path

import pytest

from src.ai import v3_refinement_difficult_check as experiment

ROOT = Path(__file__).resolve().parents[1]


def test_selection_is_tuning_only_unique_centers_order_independent_and_inside_support():
    support, rows = [], []
    for index in range(13):
        center = str(index)
        support.append(dict(center_id=center, target_log_min=[-2]*3, target_log_max=[2]*3,
                            target_centroid_normalized_log=[0]*3, target_radius_p95_normalized_log=1))
        for split, tag, thrust in [('fit', 'fit', 2.5), ('tuning', 'near', 1.2),
                                   ('tuning', 'edge', 2.0), ('tuning', 'outside', 3.0)]:
            rows.append(dict(target_id=f'{index:02d}_{tag}', source_center_id=center, internal_split=split,
                             target_performance=dict(thrust_N=thrust, tsfc_kg_N_s=1., mass_kg=1.)))
    norm = dict(log_mean=[0]*3, log_std=[1]*3)
    chosen = experiment.choose_targets(rows, support, norm)
    assert len(chosen) == len({r['source_center_id'] for r in chosen}) == 12
    assert all(r['target_id'].endswith('_edge') and r['internal_split']=='tuning' for r in chosen)
    assert chosen == experiment.choose_targets(list(reversed(rows)), list(reversed(support)), norm)
    assert chosen[0]['target_id'] == '00_edge'


def test_frozen_queue_and_code_identity_cannot_change(tmp_path, monkeypatch):
    output = tmp_path/'check'
    experiment.freeze(ROOT, output)
    lock = experiment.check_lock(ROOT, output)
    original = (output/'targets.json').read_text()
    rows = json.loads(original)
    rows.reverse()
    experiment.p.write(output/'targets.json', rows)
    with pytest.raises(ValueError, match='target queue changed'):
        experiment.check_lock(ROOT, output)
    (output/'targets.json').write_text(original)
    monkeypatch.setattr(experiment, 'dependencies', lambda _: {})
    with pytest.raises(ValueError, match='dependencies changed'):
        experiment.check_lock(ROOT, output)
    assert lock['policy']['scope'].startswith('development')


@pytest.mark.parametrize('c_hit,d_hit,control_hit,expected_trigger,expected_only', [
    (True, False, False, False, False),
    (False, True, False, True, True),
    (False, True, True, True, False),
    (False, False, True, True, False),
])
def test_stage_attribution_and_shared_c_cost(monkeypatch,c_hit,d_hit,control_hit,expected_trigger,expected_only):
    def row(hit):
        return dict(rho=.1 if hit else 2., diagnostic_domain_valid=True, software_hit_5pct=hit, feasible=False)
    c = [row(c_hit)]*8
    d = [] if c_hit else [row(d_hit)]*8
    control = dict(evaluations=[] if c_hit else [row(control_hit)]*8,
                   independent_recompute=None if c_hit else row(control_hit))
    data = {'C/evaluations.json':c,'D/evaluations.json':d,'E/evaluations.json':[],
            'F/summary.json':{'independent_recompute':row(c_hit or d_hit)},
            'target.json':{'target_id':'requested'},'B/route.json':{'status':'routable'}}
    monkeypatch.setattr(experiment.p,'read',lambda path:data[str(path)])
    report=experiment.case_report(Path('.'),control)
    assert report['D_triggered'] == expected_trigger
    assert report['D_only_rescue'] == expected_only
    assert report['production_calls_including_F'] == (9 if c_hit else 17)
    assert report['additional_control_calls_including_F'] == (0 if c_hit else 9)
