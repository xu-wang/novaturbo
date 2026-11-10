import numpy as np
from scripts.run_geometry19_framework_vs_nsga2_10x_local import local_initial, project_ball, project_discovery


def test_project_discovery_excludes_cache_rejection_and_delivery():
    summary = {'costs': {'C_calls': 1, 'D_calls': 3, 'D_cache': 2, 'D_rejections': 4, 'F_calls': 1}}
    assert project_discovery(summary) == 4


def test_local_initial_stays_in_schema_and_ball():
    center = np.full(16, 0.5); radius = 0.2
    values = local_initial(64, center, radius, np.random.default_rng(42))
    assert np.all(values >= 0) and np.all(values <= 1)
    assert np.all(np.linalg.norm(values - center, axis=1) <= radius + 1e-12)
    projected = project_ball(np.asarray([[2.0] * 16]), center, radius)
    assert np.linalg.norm(projected[0] - center) <= radius + 1e-12
