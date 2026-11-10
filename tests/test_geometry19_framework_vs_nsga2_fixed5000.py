from scripts.run_geometry19_framework_vs_nsga2_fixed5000 import task_config


def test_every_task_gets_exactly_5000_calls_independent_of_project_cost():
    config = {'fixed_physics_call_budget_per_target_per_arm': 5000}
    for calls in (1, 2, 19, 23):
        project = {'costs': {'C_calls': calls, 'F_calls': 1}}
        adapted = task_config(config, project)
        assert adapted['budget_multiplier'] * calls == 5000
