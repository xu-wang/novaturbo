"""Official non-destructive model selector for the WP19-23 pipeline."""
from __future__ import annotations

from typing import Any

MODEL_V2_1 = 'model V2.1'
MODEL_V2_2 = 'model V2.2'
MODEL_V3 = 'model V3'
MODEL_CHOICES = (MODEL_V2_1, MODEL_V2_2, MODEL_V3)


def run_wp19_23_pipeline(repository_root, *, model: str = MODEL_V2_1, **kwargs) -> dict[str, Any]:
    """Dispatch to the selected, independently versioned model entry."""
    if model == MODEL_V2_1:
        from src.ai.geometry19_pipeline import run_wp19_23_pipeline as run_v2_1
        return run_v2_1(repository_root, **kwargs)
    if model == MODEL_V2_2:
        from src.ai.geometry19_pipeline_v2_2 import run_wp19_23_pipeline_v2_2
        return run_wp19_23_pipeline_v2_2(repository_root, **kwargs)
    if model == MODEL_V3:
        if kwargs.get('pipeline_profile') == 'v3_guided_cfm_development':
            from src.ai.geometry22_pipeline_v3_guided import run_v3_routed_pipeline
            return run_v3_routed_pipeline(repository_root, **kwargs)
        if kwargs.get('pipeline_profile') == 'v3_local_cfm_development':
            from src.ai.geometry22_pipeline_v3_cfm import run_v3_routed_pipeline
            return run_v3_routed_pipeline(repository_root, **kwargs)
        if kwargs.get('pipeline_profile') == 'v3_routed_development':
            from src.ai.geometry22_pipeline_v3_routed import run_v3_routed_pipeline
            return run_v3_routed_pipeline(repository_root, **kwargs)
        from src.ai.geometry19_pipeline_v3 import run_wp19_23_pipeline_v3
        return run_wp19_23_pipeline_v3(repository_root, **kwargs)
    from src.ai.geometry19_pipeline import Geometry19PipelineError
    raise Geometry19PipelineError(f'model must be one of {MODEL_CHOICES}')
