from src.ai.geometry19_generator_failure import _metrics


def test_failure_metrics_keep_raw_prefix_semantics():
    rows=[{"failure_code":"INVALID_DISCRETE_COORDINATE","rho":None,"geometry_l2_to_reference":None},{"failure_code":None,"rho":1.5,"geometry_l2_to_reference":0.2},{"failure_code":None,"rho":0.8,"geometry_l2_to_reference":0.5}]
    result=_metrics(rows,[2,3])
    assert result["2"]["oracle_hit"] is False
    assert result["2"]["best_true_rho"]==1.5
    assert result["3"]["oracle_hit"] is True
    assert result["3"]["strict_hits"]==1
