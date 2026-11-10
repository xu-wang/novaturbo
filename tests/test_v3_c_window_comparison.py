from scripts import run_v3_c_window_comparison as comparison


def test_shared_b_ranking_extends_guided_prefix_and_nests_windows():
    raw = [dict(status='geometric_candidate', predicted_rho=float(i),
                design_id=f'd{i:03d}', raw_slot=i) for i in range(150)]
    raw += [dict(status='geometric_candidate', predicted_rho=-1.,
                 design_id='d000', raw_slot=150)]
    raw += [dict(status='outside_local_support', raw_slot=151)]
    selected = comparison.rank(raw)
    assert len(selected) == 128
    assert [r['design_id'] for r in selected[:24]] == [r['design_id'] for r in comparison.rank(raw, 24)]
    assert [r['rank'] for r in selected] == list(range(128))


def test_wide_arm_reads_shared_prefix_and_counts_valid_hits():
    rows = [dict(design_id=f'd{i}', rho=.1 if i == 30 else 2.,
                 feasible=False, diagnostic_domain_valid=i == 30,
                 software_hit_5pct=i == 30, solver_converged=True)
            for i in range(40)]
    narrow = comparison.arm(rows, 8)
    wide = comparison.arm(rows, 128)
    assert not narrow['hit'] and wide['hit']
    assert narrow['evaluations'] == 8 and wide['evaluations'] == 40
    assert wide['best_design_id'] == 'd30'
