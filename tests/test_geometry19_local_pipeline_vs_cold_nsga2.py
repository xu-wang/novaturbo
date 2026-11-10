from scripts.run_geometry19_local_pipeline_vs_cold_nsga2 import aggregate, project_counts


def test_project_counts_exclude_f_from_discovery():
    assert project_counts({'costs': {'C_calls': 1, 'D_calls': 2, 'D_cache': 4, 'F_calls': 1}}) == (3, 4)


def test_aggregate_only_compares_calls_when_both_hit():
    rows = [
        {'project_hit': True, 'cold_nsga2_hit': True, 'project_discovery_calls': 3, 'cold_nsga2_discovery_calls': 20},
        {'project_hit': True, 'cold_nsga2_hit': False, 'project_discovery_calls': 2, 'cold_nsga2_discovery_calls': 640},
    ]
    result = aggregate(rows)
    assert result['both_hit_tasks'] == 1
    assert result['both_hit_calls_saved_by_project'] == 17
    assert result['project_only_hits'] == 1
    assert result['cold_nsga2_capped_failed_calls'] == 640
