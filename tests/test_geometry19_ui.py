"""WP19-15 browser contract tests for the canonical Geometry19 workspace."""

from pathlib import Path

from ui.server import app


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_viewer_exposes_geometry19_workspace_as_primary_dashboard_entry():
    app.config.update(TESTING=True)
    response = app.test_client().get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert '<html lang="zh-CN">' in html
    assert "Geometry19 设计工作区" in html
    assert "Geometry19 · V2.1 物理计算 · 网格诊断" in html
    assert 'id="geometry19-groups"' in html
    assert 'id="geometry19-regenerate"' in html
    assert 'id="geometry19-v22-generate"' in html
    assert 'id="geometry19-model-mode"' in html
    assert "V2.1 交互设计" in html
    assert "V2.2 新设计" in html
    assert 'id="geometry19-json"' in html
    assert 'id="g19-mass"' in html
    assert "旧版设计参数" in html
    assert html.index("Geometry19 设计工作区") < html.index("T-s 图")


def test_geometry19_dashboard_uses_schema_units_validation_and_isolated_api():
    source = (REPO_ROOT / "ui" / "static" / "js" / "dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "fetch('/api/geometry19/schema')" in source
    assert "'/api/geometry19/regenerate'" in source
    assert "'/api/geometry19/solve'" in source
    assert "/api/geometry19/v2-2/jobs" in source
    assert "submitGeometry19V22Target" in source
    assert "syncGeometry19ModelMode" in source
    assert "data.schema.design_variables" in source
    assert "geometry19DisplayScale" in source
    assert "validateGeometry19Inputs" in source
    assert "parsed.best?.geometry19" in source
    assert "new CustomEvent('geometry19-model-ready'" in source
    assert "geometry19ViolationComponents" in source
    assert "performance.mass_kg" in source
    assert "operating_point?.flow_root_count" in source


def test_viewer_replaces_current_mesh_without_page_reload():
    source = (REPO_ROOT / "ui" / "static" / "js" / "viewer.js").read_text(
        encoding="utf-8"
    )

    assert "disposeCurrentComponents" in source
    assert "showGeometry19Components" in source
    assert "geometry19ComponentMetadata" in source
    assert "highlightConstraintComponents" in source
    assert "window.addEventListener('geometry19-model-ready'" in source
    assert "state.geometry19Simulation = event.detail?.simulation" in source
    assert "state.geometry19Simulation" in source
    assert "window.location.reload" not in source
