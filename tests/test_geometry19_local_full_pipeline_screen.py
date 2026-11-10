from scripts.run_geometry19_local_full_pipeline_screen import select_targets


def test_cohort_excludes_other_seeds_arms_and_rank_one_hits():
    base = dict(training_seed=42, arm='gated_top1_local_cfm', rank1_hit=False, target_index=29)
    rows = [base, dict(base, target_index=1), base,
            dict(base, target_index=2, training_seed=43),
            dict(base, target_index=3, arm='pooled_33_global_cfm'),
            dict(base, target_index=4, rank1_hit=True)]
    assert select_targets(rows) == [1, 29]
