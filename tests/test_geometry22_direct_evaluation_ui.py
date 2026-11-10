"""The browser's editable 22D form evaluates the exact submitted V3 design."""
from pathlib import Path

import pytest
import yaml

from ui.server import app


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def reference_design():
    schema = yaml.safe_load((ROOT / 'config/physics/geometry22_schema_v3_development_v1.yaml').read_text())
    return {row['name']: row['reference_value'] for row in schema['design_variables']}


def test_single_design_api_accepts_all_22_coordinates_and_reports_physics(reference_design):
    response = app.test_client().post('/api/geometry22/evaluate', json={'geometry22': reference_design})
    assert response.status_code == 200
    data = response.get_json()
    assert data['geometry22'] == reference_design and len(data['geometry22']) == 22
    assert data['schema']['prototype_dimension'] == 22
    assert data['solver']['converged'] and data['performance']['thrust_N'] > 0
    assert data['physics_contract']['physics_model_version'] == 'geometry22_map_enhanced_v3_development_v1'
    assert data['physics_evaluation_count'] == 1
    assert data['formal_use_allowed'] is False and data['deployment_allowed'] is False
    assert data['v3_physics']['compressor_total_to_total_efficiency'] > 0
    assert data['constraints']


def test_editing_v3_coordinate_changes_physical_result(reference_design):
    client = app.test_client()
    baseline = client.post('/api/geometry22/evaluate', json={'geometry22': reference_design}).get_json()
    changed = {**reference_design, 'compressor_tip_clearance_ratio': 0.004}
    response = client.post('/api/geometry22/evaluate', json={'geometry22': changed})
    assert response.status_code == 200
    edited = response.get_json()
    assert edited['design_id'] != baseline['design_id']
    assert edited['v3_physics']['compressor_total_to_total_efficiency'] < baseline['v3_physics']['compressor_total_to_total_efficiency']
    assert edited['performance']['thrust_N'] != baseline['performance']['thrust_N']


@pytest.mark.parametrize('payload', [
    {},
    {'geometry22': []},
    {'geometry22': {'unexpected': 1}},
])
def test_single_design_api_rejects_wrong_contract(payload):
    response = app.test_client().post('/api/geometry22/evaluate', json=payload)
    assert response.status_code == 400


def test_single_design_api_rejects_out_of_bounds_coordinate(reference_design):
    design = {**reference_design, 'compressor_tip_clearance_ratio': 0.01}
    response = app.test_client().post('/api/geometry22/evaluate', json={'geometry22': design})
    assert response.status_code == 400
    assert 'outside' in response.get_json()['error']
