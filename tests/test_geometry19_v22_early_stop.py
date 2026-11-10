"""Stop E within a population after the first physically accepted design."""
from src.ai.geometry19_local_full_pipeline import evaluate_until_hit


def test_e_stops_at_first_true_hit_without_evaluating_remaining_candidates():
    visited = []

    def evaluate(index):
        visited.append(index)
        return {'evaluation': {'development_diagnostic_hit': index == 3, 'index': index}}

    rows, hit = evaluate_until_hit(range(16), evaluate, True)
    assert visited == [0, 1, 2, 3]
    assert len(rows) == 4
    assert hit['index'] == 3


def test_e_old_population_behavior_remains_available_for_frozen_protocols():
    visited = []

    def evaluate(index):
        visited.append(index)
        return {'evaluation': {'development_diagnostic_hit': index == 3, 'index': index}}

    rows, hit = evaluate_until_hit(range(16), evaluate)
    assert len(visited) == len(rows) == 16
    assert hit['index'] == 3
