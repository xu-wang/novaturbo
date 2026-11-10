from scripts.run_geometry19_local_pipeline_cost_comparison import aggregate, summarize_target


def evaluation(hit):
    return {'development_diagnostic_hit': hit}


def test_costs_use_first_hit_but_coverage_uses_all_candidates():
    enhanced = {'first_hit_stage': 'D', 'costs': {'C_calls': 1, 'D_calls': 2, 'F_calls': 1}}
    row = summarize_target(5, [evaluation(False), evaluation(True), evaluation(True)], enhanced)
    assert row['ranked_candidates_evaluated'] == 3
    assert row['missing_slots'] == 61
    assert row['first_hit_rank'] == 2
    assert row['hit_count'] == 2
    assert row['sequential_delivery_calls'] == 3
    assert row['enhanced_delivery_calls'] == 4
    assert row['delivery_call_delta_enhanced_minus_sequential'] == 1


def test_aggregate_reports_paired_wins_and_reduction():
    rows = [
        {'first_hit_rank': 1, 'ranked_candidates_evaluated': 64, 'missing_slots': 0, 'sequential_delivery_calls': 2, 'enhanced_delivery_calls': 2},
        {'first_hit_rank': 5, 'ranked_candidates_evaluated': 60, 'missing_slots': 4, 'sequential_delivery_calls': 6, 'enhanced_delivery_calls': 4},
        {'first_hit_rank': 2, 'ranked_candidates_evaluated': 17, 'missing_slots': 47, 'sequential_delivery_calls': 3, 'enhanced_delivery_calls': 5},
    ]
    result = aggregate(rows)
    assert result['paired_wins'] == {'tie': 1, 'enhanced': 1, 'sequential': 1}
    assert result['sequential_delivery_calls'] == result['enhanced_delivery_calls'] == 11
    assert result['all_available_candidates_evaluated']
