#!/usr/bin/env python3
"""Build the target-directed inverse-design result presentation.

All values and example designs are loaded from the frozen 10D benchmark.  The
engine cross-sections are produced with NovaTurbo's parametric geometry module.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.geometry.assembly import EngineAssemblyParams, assemble_engine
from src.export.stl_export import (
    _create_cylinder,
    _create_transitions,
    export_combustor_stl,
    export_compressor_stl,
    export_inlet_stl,
    export_nozzle_stl,
    export_turbine_stl,
)


RESULT_ROOT = REPO_ROOT / "data/inverse_design/target_directed_active10d_gpu6"
OUTPUT_DIR = REPO_ROOT / "docs/presentations"
ASSET_DIR = OUTPUT_DIR / "target_inverse_design_assets"
OUTPUT_PPTX = OUTPUT_DIR / "生成式模型目标导向反向设计结果汇报.pptx"

FONT = "Noto Sans CJK SC"
BG = "07111F"
PANEL = "0E1D2E"
PANEL_2 = "14273B"
WHITE = "F4F8FC"
MUTED = "A9BCD0"
CYAN = "25D0E8"
GREEN = "57D9A3"
CORAL = "FF786F"
AMBER = "FFC857"
BLUE = "5B8FF9"
PURPLE = "9A7FF0"
GRID = "2A4056"

METHOD_LABELS = {
    "random": "Bare Random",
    "screened_random": "Screened Random",
    "cvae": "cVAE",
    "nflows": "nflows",
    "cinn": "cINN",
    "cfm": "CFM",
}
METHOD_COLORS = {
    "random": "#8799AB",
    "screened_random": "#FF786F",
    "cvae": "#FFC857",
    "nflows": "#25D0E8",
    "cinn": "#57D9A3",
    "cfm": "#9A7FF0",
}


def _rgb(value: str) -> RGBColor:
    value = value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Droid Sans Fallback", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": f"#{BG}",
            "axes.facecolor": f"#{PANEL}",
            "axes.edgecolor": f"#{GRID}",
            "axes.labelcolor": f"#{MUTED}",
            "xtick.color": f"#{MUTED}",
            "ytick.color": f"#{MUTED}",
            "text.color": f"#{WHITE}",
            "grid.color": f"#{GRID}",
            "grid.alpha": 0.55,
        }
    )


def _save_figure(fig: plt.Figure, name: str) -> Path:
    path = ASSET_DIR / name
    fig.savefig(path, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _load_timing() -> pd.DataFrame:
    rows = []
    for run_dir in (RESULT_ROOT / "runs").iterdir():
        path = run_dir / "summary/metrics.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        method = value.get("model_id") or value["mode"]
        rows.append({"method": method, **value["cost"]})
    return pd.DataFrame(rows)


def _best_initial_record(target_id: str, method: str) -> pd.Series:
    run_id = f"target_t42_{target_id[:10]}_s42_{method}"
    ledger = pd.read_csv(RESULT_ROOT / "runs" / run_id / "optimization/evaluation_ledger.csv")
    initial = ledger.loc[
        (ledger["generation"].astype(int) == 0) & ledger["is_feasible"].astype(bool)
    ]
    return initial.sort_values("objective_internal_0", kind="stable").iloc[0]


def _design_from_record(record: pd.Series) -> Dict[str, float]:
    return {
        column.removeprefix("gene__"): float(record[column])
        for column in record.index
        if column.startswith("gene__")
    }


def _assembly_from_design(design: Mapping[str, float]):
    params = EngineAssemblyParams()
    diameter = float(design["compressor_diameter_mm"])
    params.compressor.impeller_tip_diameter_mm = diameter
    params.compressor.impeller_hub_diameter_mm = 0.25 * diameter
    params.compressor.inducer_tip_diameter_mm = 0.625 * diameter
    params.compressor.blade_count = int(round(design["compressor_blade_count"]))
    params.compressor.splitter_blade_count = params.compressor.blade_count
    params.compressor.pressure_ratio = float(design["compressor_pressure_ratio"])
    params.compressor.mass_flow_kg_s = float(design["mass_flow_kg_s"])
    params.compressor.rpm = float(design["rpm"])

    params.inlet.inlet_diameter_mm = diameter
    params.inlet.outlet_diameter_mm = 0.625 * diameter

    params.combustor.length_mm = float(design["combustor_length_mm"])
    params.combustor.outer_diameter_mm = float(design["combustor_outer_diameter_mm"])
    params.combustor.inner_diameter_mm = min(
        float(design["combustor_inner_diameter_mm"]),
        params.combustor.outer_diameter_mm - 8.0,
    )
    params.combustor.liner_thickness_mm = float(
        design["combustor_liner_thickness_mm"]
    )
    params.combustor.num_fuel_injectors = int(
        round(design["combustor_num_injectors"])
    )
    params.combustor.air_fuel_ratio = float(design["combustor_air_fuel_ratio"])
    params.combustor.outlet_temperature_K = float(design["turbine_inlet_temp_K"])

    turbine_tip = 1.05 * diameter
    hub_ratio = float(design["turbine_hub_tip_ratio"])
    params.turbine.tip_diameter_mm = turbine_tip
    params.turbine.hub_tip_ratio = hub_ratio
    params.turbine.hub_diameter_mm = turbine_tip * hub_ratio
    params.turbine.blade_height_mm = 0.5 * (
        turbine_tip - params.turbine.hub_diameter_mm
    )
    params.turbine.blade_count = int(round(design["turbine_blade_count"]))
    params.turbine.rpm = float(design["rpm"])
    params.turbine.inlet_temperature_K = float(design["turbine_inlet_temp_K"])

    params.nozzle.exit_diameter_mm = float(design["nozzle_exit_diameter_mm"])
    params.nozzle.inlet_diameter_mm = max(
        turbine_tip, params.nozzle.exit_diameter_mm + 5.0
    )
    params.nozzle.flange_outer_diameter_mm = params.nozzle.inlet_diameter_mm + 10.0
    params.nozzle.bolt_circle_diameter_mm = params.nozzle.inlet_diameter_mm + 5.0
    return params, assemble_engine(params)


def _profile_arrays(geometry, position: float, component: str):
    if component == "inlet":
        outer = geometry.outer_contour
        inner = geometry.inner_contour
    elif component == "compressor":
        outer = geometry.shroud_contour
        inner = geometry.hub_contour
    elif component == "combustor":
        outer = geometry.outer_liner_outer
        inner = geometry.inner_liner_inner
    elif component == "turbine":
        outer = geometry.casing_contour
        inner = geometry.hub_contour
    elif component == "nozzle":
        outer = geometry.outer_contour
        inner = geometry.inner_contour
    else:
        raise ValueError(component)
    z_min = max(float(np.min(outer[:, 0])), float(np.min(inner[:, 0])))
    z_max = min(float(np.max(outer[:, 0])), float(np.max(inner[:, 0])))
    z = np.linspace(z_min, z_max, 100)
    r_outer = np.interp(z, outer[:, 0], outer[:, 1])
    r_inner = np.interp(z, inner[:, 0], inner[:, 1])
    lo = np.minimum(r_inner, r_outer)
    hi = np.maximum(r_inner, r_outer)
    return z + position, lo, hi


def _plot_engine_cross_section(
    design: Mapping[str, float], title: str, path_name: str
) -> Path:
    params, asm = _assembly_from_design(design)
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    components = [
        ("inlet", asm.inlet_geo, "#4FC3F7", "进气道"),
        ("compressor", asm.compressor_geo, "#57D9A3", "压气机"),
        ("combustor", asm.combustor_geo, "#FF8A65", "燃烧室"),
        ("turbine", asm.turbine_geo, "#FFC857", "涡轮"),
        ("nozzle", asm.nozzle_geo, "#FF786F", "喷管"),
    ]
    label_lift = {
        "inlet": 7.0,
        "compressor": 20.0,
        "combustor": 7.0,
        "turbine": 19.0,
        "nozzle": 7.0,
    }
    for component, geometry, color, label in components:
        position = asm.component_positions[component]
        z, inner, outer = _profile_arrays(geometry, position, component)
        ax.fill_between(z, inner, outer, color=color, alpha=0.88, linewidth=0)
        ax.fill_between(z, -outer, -inner, color=color, alpha=0.88, linewidth=0)
        ax.plot(z, outer, color=color, lw=1.2)
        ax.plot(z, -outer, color=color, lw=1.2)
        ax.text(
            float(np.mean(z)),
            float(np.max(outer) + label_lift[component]),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
    ax.axhline(0.0, color=f"#{MUTED}", lw=0.7, alpha=0.55)
    # A schematic axial view is deliberately used here: preserving the exact
    # radial/axial aspect would make compact micro-engine differences unreadable
    # on a presentation slide.
    ax.set_aspect("auto")
    ax.set_xlim(-2, asm.total_length_mm + 4)
    max_r = max(asm.max_diameter_mm / 2.0 + 20.0, 75.0)
    ax.set_ylim(-max_r, max_r)
    ax.set_xlabel("轴向位置 / mm")
    ax.set_ylabel("半径 / mm")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    fig.tight_layout()
    return _save_figure(fig, path_name)


def _mesh_face_colors(triangles: np.ndarray, color: str) -> np.ndarray:
    """Apply deterministic directional lighting to a rendered STL surface."""

    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(first, second)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(length, 1.0e-12)
    light = np.asarray([0.35, -0.45, 0.82], dtype=float)
    light /= np.linalg.norm(light)
    # Keep enough directional shading to reveal the 3D shape without making
    # the individual STL triangles dominate the presentation rendering.
    intensity = 0.70 + 0.30 * np.clip(normals @ light, 0.0, 1.0)
    base = np.asarray(to_rgb(color), dtype=float)
    rgb = np.clip(intensity[:, None] * base[None, :], 0.0, 1.0)
    return np.column_stack([rgb, np.full(len(rgb), 0.98)])


def _render_engine_3d(
    design: Mapping[str, float], title: str, path_name: str
) -> Path:
    """Render actual NovaTurbo component meshes in an assembled 3D view."""

    params, asm = _assembly_from_design(design)
    positions = asm.component_positions
    # Lower the circumferential tessellation at export time for presentation
    # rendering.  Keeping every resulting triangle avoids the holes introduced
    # by random face dropping while retaining the true NovaTurbo geometry.
    n_theta = 20
    meshes = [
        (
            "进气道",
            export_inlet_stl(
                asm.inlet_geo, params.inlet, positions["inlet"], n_theta
            ),
            "#4FC3F7",
        ),
        (
            "压气机",
            export_compressor_stl(
                asm.compressor_geo,
                params.compressor,
                positions["compressor"],
                n_theta,
            ),
            "#57D9A3",
        ),
        (
            "燃烧室",
            export_combustor_stl(
                asm.combustor_geo,
                params.combustor,
                positions["combustor"],
                n_theta,
            ),
            "#FF8A65",
        ),
        (
            "涡轮",
            export_turbine_stl(
                asm.turbine_geo,
                params.turbine,
                positions["turbine"],
                n_theta,
            ),
            "#FFC857",
        ),
        (
            "喷管",
            export_nozzle_stl(
                asm.nozzle_geo, params.nozzle, positions["nozzle"], n_theta
            ),
            "#FF786F",
        ),
        (
            "轴",
            _create_cylinder(
                0.0, asm.total_length_mm, params.shaft_diameter_mm / 2.0, n_theta
            ),
            "#B0BEC5",
        ),
        ("过渡段", _create_transitions(asm, params, n_theta), "#9A7FF0"),
    ]

    fig = plt.figure(figsize=(10.8, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    all_vertices = []
    for _label, mesh, color in meshes:
        # STL uses the Z axis as the engine axis.  Rotate it onto plot X so the
        # assembled engine reads naturally from inlet to nozzle.
        vertices = np.column_stack(
            [mesh.vertices[:, 2], mesh.vertices[:, 0], mesh.vertices[:, 1]]
        )
        faces = np.asarray(mesh.faces, dtype=int)
        triangles = vertices[faces]
        collection = Poly3DCollection(
            triangles,
            facecolors=_mesh_face_colors(triangles, color),
            edgecolors="none",
            linewidths=0.0,
        )
        ax.add_collection3d(collection)
        all_vertices.append(vertices)

    points = np.vstack(all_vertices)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    span = maximum - minimum
    padding = np.asarray([0.04, 0.12, 0.12]) * np.maximum(span, 1.0)
    ax.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    ax.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
    ax.set_zlim(minimum[2] - padding[2], maximum[2] + padding[2])
    ax.set_box_aspect(
        (span[0], max(span[1], 1.0), max(span[2], 1.0)), zoom=1.48
    )
    ax.view_init(elev=18, azim=-72)
    ax.set_axis_off()
    ax.set_facecolor(f"#{BG}")
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=1)
    legend = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, _mesh, color in meshes[:5]
    ]
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        labelcolor=f"#{WHITE}",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.04)
    return _save_figure(fig, path_name)


def _make_charts(aggregate: pd.DataFrame, metrics: pd.DataFrame, convergence: pd.DataFrame):
    order = ["random", "screened_random", "cvae", "nflows", "cinn", "cfm"]
    data = aggregate.set_index("method").loc[order]

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    x = np.arange(len(order))
    width = 0.36
    ax.bar(
        x - width / 2,
        data["initial_best_error_mean"],
        width,
        color=[METHOD_COLORS[m] for m in order],
        label="初始最佳误差",
    )
    ax.bar(
        x + width / 2,
        data["initial_top10_error_mean"],
        width,
        facecolor="none",
        edgecolor=[METHOD_COLORS[m] for m in order],
        linewidth=2,
        label="初始 Top-10 平均误差",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, [METHOD_LABELS[m] for m in order])
    ax.set_ylabel("等权相对目标误差 J（对数尺度，越低越好）")
    ax.set_title("生成器把 generation-0 误差降低 1–2 个数量级", loc="left", fontweight="bold")
    ax.grid(axis="y", which="both", linestyle="--")
    ax.legend(frameon=False, labelcolor=f"#{WHITE}", ncol=2, loc="upper right")
    for index, value in enumerate(data["initial_best_error_mean"]):
        ax.text(index - width / 2, value * 1.35, f"{value:.2g}", ha="center", fontsize=9)
    fig.tight_layout()
    accuracy = _save_figure(fig, "accuracy_comparison.png")

    grouped = metrics.groupby("method").mean(numeric_only=True).loc[order]
    successful_eval = grouped["search_evaluations_to_hit"].copy()
    conservative = grouped["prefilter_evaluations"] + successful_eval
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.6))
    axes[0].bar(
        np.arange(len(order)),
        grouped["initial_hit_rate"] * 100.0,
        color=[METHOD_COLORS[m] for m in order],
    )
    axes[0].set_xticks(np.arange(len(order)), [METHOD_LABELS[m] for m in order], rotation=18)
    axes[0].set_ylabel("初始种群命中率 / %")
    axes[0].set_ylim(0, 108)
    axes[0].grid(axis="y", linestyle="--")
    axes[0].set_title("40个初始设计中有多少已经可用", loc="left", fontweight="bold")
    for i, value in enumerate(grouped["initial_hit_rate"] * 100.0):
        axes[0].text(i, value + 3, f"{value:.0f}%", ha="center", fontsize=9)
    axes[1].bar(
        np.arange(len(order)),
        conservative,
        color=[METHOD_COLORS[m] for m in order],
    )
    axes[1].set_xticks(np.arange(len(order)), [METHOD_LABELS[m] for m in order], rotation=18)
    axes[1].set_ylabel("首次命中前物理评估次数（成功运行均值）")
    axes[1].grid(axis="y", linestyle="--")
    axes[1].set_title("目标解更早出现", loc="left", fontweight="bold")
    for i, value in enumerate(conservative):
        axes[1].text(i, value + 12, f"{value:.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    hit_efficiency = _save_figure(fig, "hit_and_evaluations.png")

    selected_methods = ["random", "screened_random", "nflows", "cinn"]
    curve = (
        convergence.loc[convergence["method"].isin(selected_methods)]
        .groupby(["method", "generation"], as_index=False)["best_error"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    for method in selected_methods:
        part = curve.loc[curve["method"] == method]
        ax.plot(
            part["generation"],
            part["best_error"],
            marker="o",
            markersize=3.2,
            linewidth=2.2,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax.set_yscale("log")
    ax.set_xlabel("遗传搜索代数")
    ax.set_ylabel("累计最佳目标误差 J（越低越好）")
    ax.set_title("生成器从第0代开始领先，15代后优势仍保留", loc="left", fontweight="bold")
    ax.grid(which="both", linestyle="--")
    ax.legend(frameon=False, labelcolor=f"#{WHITE}", ncol=2)
    fig.tight_layout()
    convergence_path = _save_figure(fig, "convergence.png")

    return accuracy, hit_efficiency, convergence_path


def _blank_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = _rgb(BG)
    return slide


def _add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 20,
    color: str = WHITE,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin: float = 0.03,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = _rgb(color)
    return box


def _add_rich_lines(slide, lines: Sequence[tuple[str, str, bool]], x, y, w, h, size=19):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.06)
    frame.margin_right = Inches(0.06)
    for index, (text, color, bold) in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(9)
        run = paragraph.add_run()
        run.text = text
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = _rgb(color)
    return box


def _panel(slide, x, y, w, h, color=PANEL, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(color)
    shape.line.color.rgb = _rgb(color)
    return shape


def _title(slide, kicker: str, title: str, number: int):
    _add_text(slide, kicker.upper(), 0.55, 0.27, 8.5, 0.3, size=10, color=CYAN, bold=True)
    _add_text(slide, title, 0.55, 0.62, 11.9, 0.65, size=27, bold=True)
    _add_text(slide, f"{number:02d}", 12.25, 0.31, 0.5, 0.3, size=11, color=MUTED, bold=True, align=PP_ALIGN.RIGHT)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.56), Inches(1.28), Inches(12.2), Inches(0.015))
    line.fill.solid(); line.fill.fore_color.rgb = _rgb(GRID); line.line.fill.background()


def _metric(slide, value: str, label: str, x, y, w, accent=CYAN):
    _panel(slide, x, y, w, 1.05, PANEL)
    _add_text(slide, value, x + 0.18, y + 0.12, w - 0.36, 0.43, size=23, color=accent, bold=True)
    _add_text(slide, label, x + 0.18, y + 0.62, w - 0.36, 0.25, size=10.5, color=MUTED)


def _bullet_lines(slide, lines: Iterable[str], x, y, w, h, size=17, color=WHITE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear(); frame.word_wrap = True
    for index, text in enumerate(lines):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = text
        p.level = 0
        p.space_after = Pt(10)
        p.font.name = FONT; p.font.size = Pt(size); p.font.color.rgb = _rgb(color)
        p.text = f"•  {text}"
    return box


def _add_picture(slide, path: Path, x, y, w, h):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def _add_table(slide, rows: Sequence[Sequence[str]], x, y, w, h, widths=None):
    table = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if widths:
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.margin_left = Inches(0.07); cell.margin_right = Inches(0.07)
            cell.margin_top = Inches(0.04); cell.margin_bottom = Inches(0.04)
            cell.fill.solid(); cell.fill.fore_color.rgb = _rgb(PANEL_2 if r == 0 else PANEL)
            cell.border if False else None
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.name = FONT
                    run.font.size = Pt(11.5 if r else 11)
                    run.font.bold = r == 0
                    run.font.color.rgb = _rgb(CYAN if r == 0 else WHITE)
    return table


def _case_values(target: pd.Series, method_records: Mapping[str, pd.Series]):
    values = []
    for metric, performance_column, scale in [
        ("推力 / N", "performance__thrust_N", 1.0),
        ("TSFC / g·kN⁻¹·s⁻¹", "performance__tsfc_kg_N_s", 1.0e6),
        ("质量 / kg", "performance__total_mass_kg", 1.0),
    ]:
        target_name = {
            "performance__thrust_N": "thrust_N",
            "performance__tsfc_kg_N_s": "tsfc_kg_N_s",
            "performance__total_mass_kg": "total_mass_kg",
        }[performance_column]
        row = [metric, f"{float(target[target_name]) * scale:.4g}"]
        for method in ("nflows", "screened_random"):
            row.append(f"{float(method_records[method][performance_column]) * scale:.4g}")
        values.append(row)
    return values


def build_presentation() -> Path:
    _configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    aggregate = pd.read_csv(RESULT_ROOT / "aggregate.csv")
    metrics = pd.read_csv(RESULT_ROOT / "run_metrics.csv")
    convergence = pd.read_csv(RESULT_ROOT / "convergence.csv")
    targets = pd.read_csv(RESULT_ROOT / "selected_targets.csv")
    timing = _load_timing().groupby("method").mean(numeric_only=True)
    accuracy_chart, hit_chart, convergence_chart = _make_charts(
        aggregate, metrics, convergence
    )

    case_targets = [targets.iloc[0], targets.iloc[2]]
    case_assets = []
    for case_index, target in enumerate(case_targets, start=1):
        target_id = str(target["target_id"])
        records = {
            method: _best_initial_record(target_id, method)
            for method in ("nflows", "screened_random")
        }
        images = {
            method: _render_engine_3d(
                _design_from_record(record),
                f"{METHOD_LABELS[method]} 最佳初始设计 · NovaTurbo 3D",
                f"case_{case_index}_{method}_geometry.png",
            )
            for method, record in records.items()
        }
        case_assets.append((target, records, images))

    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)

    # 1 — Title
    slide = _blank_slide(prs)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.16), Inches(7.5))
    accent.fill.solid(); accent.fill.fore_color.rgb = _rgb(CYAN); accent.line.fill.background()
    _add_text(slide, "NOVATURBO · TARGET-DIRECTED INVERSE DESIGN", 0.72, 0.65, 9.8, 0.35, size=12, color=CYAN, bold=True)
    _add_text(slide, "生成式模型用于\n给定目标性能的反向设计", 0.72, 1.35, 8.7, 1.75, size=38, bold=True)
    _add_text(slide, "从“随机搜索”转向“直接生成目标附近的高质量初始设计”", 0.75, 3.38, 8.8, 0.5, size=18, color=MUTED)
    _metric(slide, "96 / 96", "正式目标导向运行完成", 0.75, 4.55, 2.35, CYAN)
    _metric(slide, "10D → 19D", "有效变量生成与完整设计接口", 3.28, 4.55, 2.35, GREEN)
    _metric(slide, "GPU 6", "训练与正式生成环境", 5.81, 4.55, 2.1, AMBER)
    _panel(slide, 9.55, 0.75, 3.05, 5.65, PANEL)
    _add_text(slide, "核心结论", 9.9, 1.12, 2.3, 0.35, size=14, color=CYAN, bold=True)
    _add_text(slide, "204.9×", 9.88, 1.78, 2.3, 0.62, size=33, color=GREEN, bold=True)
    _add_text(slide, "nflows 相对同预算\nScreened Random 的\n初始最佳误差改善", 9.9, 2.43, 2.25, 1.25, size=16, color=WHITE)
    _add_text(slide, "100%", 9.88, 4.10, 2.3, 0.57, size=31, color=CYAN, bold=True)
    _add_text(slide, "nflows / cINN\n初始种群命中率", 9.9, 4.72, 2.25, 0.8, size=16, color=WHITE)
    _add_text(slide, "2026-08-11", 0.75, 6.85, 2.0, 0.25, size=10, color=MUTED)

    # 2 — Question
    slide = _blank_slide(prs); _title(slide, "01 · RESEARCH QUESTION", "我们真正要回答的是什么？", 2)
    _panel(slide, 0.6, 1.65, 5.75, 4.75, PANEL)
    _add_text(slide, "错误评价口径", 0.95, 2.02, 2.0, 0.35, size=15, color=CORAL, bold=True)
    _add_text(slide, "最大化全局 Pareto Hypervolume", 0.95, 2.65, 4.8, 0.48, size=22, bold=True)
    _add_text(slide, "max F  ·  min TSFC  ·  min Mass  ·  max T/W", 0.95, 3.28, 4.8, 0.35, size=14, color=MUTED)
    _add_text(slide, "它奖励前沿覆盖，而不是接近一个给定性能点。", 0.95, 4.15, 4.8, 0.7, size=17, color=WHITE)
    _add_text(slide, "不作为本研究主指标", 0.95, 5.45, 3.0, 0.35, size=15, color=CORAL, bold=True)
    _panel(slide, 6.75, 1.65, 5.95, 4.75, PANEL_2)
    _add_text(slide, "正确评价口径", 7.1, 2.02, 2.0, 0.35, size=15, color=GREEN, bold=True)
    _add_text(slide, "给定目标，谁的初始设计最接近？", 7.1, 2.62, 5.0, 0.5, size=22, bold=True)
    _add_text(slide, "J(x) = ⅓ (eF + eTSFC + eMass)", 7.1, 3.44, 4.8, 0.5, size=24, color=CYAN, bold=True)
    _bullet_lines(slide, ["J = 0 表示物理回代精确命中", "主指标：初始最佳误差、Top-K、命中率", "搜索目标：min J，而非原始目标取极值"], 7.1, 4.25, 4.8, 1.55, size=15)

    # 3 — Pipeline
    slide = _blank_slide(prs); _title(slide, "02 · METHOD", "生成式 Warm-Start：从性能目标直接到设计", 3)
    steps = [
        ("目标条件", "F* · TSFC* · Mass*", CYAN),
        ("条件生成器", "cVAE / nflows / cINN / CFM", GREEN),
        ("10D设计", "全部当前物理有效变量", AMBER),
        ("Physics-in-loop", "边界 + 0D + 质量回代", CORAL),
        ("Top-40", "按目标误差 J 选择", PURPLE),
        ("目标导向搜索", "15代 · min J", BLUE),
    ]
    start_x = 0.55
    for i, (head, body, color) in enumerate(steps):
        x = start_x + i * 2.1
        _panel(slide, x, 2.15, 1.75, 2.4, PANEL)
        _add_text(slide, f"{i+1:02d}", x + 0.15, 2.35, 0.45, 0.3, size=11, color=color, bold=True)
        _add_text(slide, head, x + 0.15, 2.85, 1.45, 0.55, size=16, color=WHITE, bold=True)
        _add_text(slide, body, x + 0.15, 3.55, 1.45, 0.65, size=11.5, color=MUTED)
        if i < len(steps) - 1:
            _add_text(slide, "→", x + 1.78, 3.02, 0.34, 0.4, size=21, color=GRID, bold=True, align=PP_ALIGN.CENTER)
    _panel(slide, 0.75, 5.25, 11.9, 0.85, PANEL_2)
    _add_text(slide, "公平性原则", 1.02, 5.49, 1.25, 0.28, size=12, color=CYAN, bold=True)
    _add_text(slide, "Generator 与 Screened Random：相同256次候选机会、相同物理回代、相同Top-K、相同40×15搜索预算", 2.35, 5.43, 9.8, 0.35, size=14, color=WHITE)

    # 4 — Experiment
    slide = _blank_slide(prs); _title(slide, "03 · EXPERIMENT", "96次正式运行：比较的是初始化方法，不是运气", 4)
    _metric(slide, "4", "目标：2 ID + 2 Interpolation", 0.7, 1.7, 2.25, CYAN)
    _metric(slide, "2 × 2", "训练种子 × 优化种子", 3.15, 1.7, 2.25, GREEN)
    _metric(slide, "6", "Random / Screened / 四类Generator", 5.6, 1.7, 2.55, AMBER)
    _metric(slide, "96 / 96", "运行完成且账本一致", 8.35, 1.7, 2.25, PURPLE)
    _metric(slide, "137", "自动测试全部通过", 10.8, 1.7, 1.8, BLUE)
    _panel(slide, 0.7, 3.15, 5.85, 2.7, PANEL)
    _add_text(slide, "初始化对照", 1.0, 3.48, 2.0, 0.35, size=16, color=CYAN, bold=True)
    _bullet_lines(slide, ["Bare Random：直接40个随机19D", "Screened Random：256个随机10D → 物理筛选 → Top-40", "Generator：256个条件10D → 同一物理筛选 → Top-40"], 1.0, 4.05, 5.0, 1.45, size=14)
    _panel(slide, 6.8, 3.15, 5.8, 2.7, PANEL)
    _add_text(slide, "报告指标", 7.1, 3.48, 2.0, 0.35, size=16, color=GREEN, bold=True)
    _bullet_lines(slide, ["初始最佳 J 与 Top-10 平均 J", "40个初始个体中三条件同时命中的比例", "达到阈值的物理评估次数与15代后最佳 J"], 7.1, 4.05, 4.95, 1.45, size=14)

    # 5 — Accuracy
    slide = _blank_slide(prs); _title(slide, "04 · ACCURACY", "生成器把初始误差降低1–2个数量级", 5)
    _add_picture(slide, accuracy_chart, 0.55, 1.55, 8.25, 4.85)
    _panel(slide, 9.05, 1.68, 3.65, 4.55, PANEL)
    _add_text(slide, "相对 Screened Random", 9.38, 2.02, 2.9, 0.35, size=13, color=MUTED, bold=True)
    _add_text(slide, "204.9×", 9.35, 2.55, 2.8, 0.62, size=32, color=CYAN, bold=True)
    _add_text(slide, "nflows 初始最佳误差改善", 9.38, 3.16, 2.7, 0.35, size=13, color=WHITE)
    _add_text(slide, "184.1×", 9.35, 3.85, 2.8, 0.55, size=28, color=GREEN, bold=True)
    _add_text(slide, "cINN 初始最佳误差改善", 9.38, 4.42, 2.7, 0.35, size=13, color=WHITE)
    _add_text(slide, "16 / 16", 9.35, 5.03, 2.8, 0.5, size=25, color=AMBER, bold=True)
    _add_text(slide, "四种生成器均赢得全部初始误差配对", 9.38, 5.54, 2.7, 0.5, size=12.5, color=WHITE)

    # 6 — Hit rate / evaluation count
    slide = _blank_slide(prs); _title(slide, "05 · EFFICIENCY", "优势首先体现为更少评估得到可用设计", 6)
    _add_picture(slide, hit_chart, 0.55, 1.48, 8.2, 4.9)
    _panel(slide, 8.95, 1.65, 3.75, 4.6, PANEL)
    _add_text(slide, "首次命中的保守评估账本", 9.28, 1.98, 3.0, 0.4, size=14, color=CYAN, bold=True)
    _add_text(slide, "≈129", 9.25, 2.63, 2.4, 0.6, size=31, color=GREEN, bold=True)
    _add_text(slide, "nflows：预筛有效评估 + 第1个初始个体", 9.28, 3.2, 2.85, 0.62, size=12.5)
    _add_text(slide, "≈493", 9.25, 4.02, 2.4, 0.55, size=29, color=CORAL, bold=True)
    _add_text(slide, "Screened Random：仅统计最终成功的运行", 9.28, 4.58, 2.85, 0.58, size=12.5)
    _add_text(slide, "3.8× 更少", 9.25, 5.42, 2.7, 0.45, size=22, color=CYAN, bold=True)
    _add_text(slide, "达到同一冻结容差", 9.28, 5.88, 2.7, 0.28, size=11.5, color=MUTED)

    # 7 — Speed truthfully
    slide = _blank_slide(prs); _title(slide, "06 · SPEED", "当前0D场景下，网络采样已足够快；核心收益是减少搜索", 7)
    nflows_time = float(timing.loc["nflows", "generator_sampling_seconds"])
    screen_wall = float(timing.loc["screened_random", "current_execution_wall_seconds"])
    nflows_wall = float(timing.loc["nflows", "current_execution_wall_seconds"])
    _metric(slide, f"{nflows_time*1000:.0f} ms", "nflows生成256个10D候选", 0.75, 1.75, 2.65, CYAN)
    _metric(slide, f"{nflows_wall:.2f} s", "nflows完整15代运行均值", 3.65, 1.75, 2.65, GREEN)
    _metric(slide, f"{screen_wall:.2f} s", "Screened Random完整15代均值", 6.55, 1.75, 2.65, CORAL)
    _metric(slide, "0代", "nflows/cINN已经得到命中解", 9.45, 1.75, 2.65, AMBER)
    _panel(slide, 0.75, 3.25, 11.35, 2.4, PANEL)
    _add_text(slide, "如何理解“速度优势”", 1.08, 3.58, 3.0, 0.35, size=17, color=CYAN, bold=True)
    _bullet_lines(slide, ["当前Brayton 0D单次评价很便宜，因此原始墙钟时间只相差约3%，不应夸大。", "生成器的确定优势是：在搜索开始前即提供命中解，显著减少达到阈值所需的物理评价。", "当0D替换为高保真CFD/FEA时，少约3.8倍评估将直接转化为更大的实际时间与计算成本优势。"], 1.08, 4.12, 10.4, 1.25, size=14.5)

    # 8 — Population quality
    slide = _blank_slide(prs); _title(slide, "07 · POPULATION QUALITY", "不是“偶然找到一个点”，而是一组目标附近的设计", 8)
    rows = [["方法", "候选池命中率", "初始40个体命中率", "至少1个命中解"]]
    agg = aggregate.set_index("method")
    for method in ["random", "screened_random", "cvae", "nflows", "cinn", "cfm"]:
        row = agg.loc[method]
        candidate = "—" if pd.isna(row["candidate_hit_rate_mean"]) else f"{row['candidate_hit_rate_mean']:.1%}"
        rows.append([METHOD_LABELS[method], candidate, f"{row['initial_hit_rate_mean']:.1%}", f"{row['initial_target_success_rate']:.0%}"])
    _add_table(slide, rows, 0.72, 1.7, 7.35, 4.35, widths=[2.2, 1.75, 1.8, 1.6])
    _panel(slide, 8.45, 1.7, 4.15, 4.35, PANEL)
    _add_text(slide, "nflows / cINN", 8.82, 2.08, 3.25, 0.45, size=24, color=CYAN, bold=True)
    _add_text(slide, "初始40个设计全部满足\n推力、TSFC、质量三项冻结容差", 8.82, 2.78, 3.25, 1.05, size=18, bold=True)
    _add_text(slide, "意味着：", 8.82, 4.15, 1.0, 0.3, size=13, color=MUTED, bold=True)
    _bullet_lines(slide, ["可直接交付多个候选", "保留反问题的一对多解", "降低后续局部搜索风险"], 8.82, 4.58, 2.95, 1.15, size=14)

    # 9–10 — Cases
    for slide_index, (target, records, images) in enumerate(case_assets, start=9):
        case_name = "ID目标：训练分布内已知可行性能" if target["target_type"] == "ID" else "Interpolation目标：未见性能点插值"
        slide = _blank_slide(prs); _title(slide, f"0{slide_index-1} · NOVATURBO CASE", case_name, slide_index)
        _add_picture(slide, images["nflows"], 0.55, 1.5, 6.05, 2.25)
        _add_picture(slide, images["screened_random"], 6.75, 1.5, 6.05, 2.25)
        table_rows = [["性能", "目标", "nflows", "Screened Random"], *_case_values(target, records)]
        _add_table(slide, table_rows, 0.72, 4.05, 7.45, 1.9, widths=[2.4, 1.55, 1.55, 1.95])
        nflow_error = float(records["nflows"]["objective_internal_0"])
        random_error = float(records["screened_random"]["objective_internal_0"])
        _panel(slide, 8.5, 4.05, 4.1, 1.9, PANEL)
        _add_text(slide, "综合目标误差 J", 8.83, 4.34, 2.6, 0.3, size=13, color=MUTED, bold=True)
        _add_text(slide, f"{nflow_error:.6f}", 8.82, 4.78, 2.4, 0.42, size=22, color=CYAN, bold=True)
        _add_text(slide, "nflows", 11.18, 4.85, 0.9, 0.25, size=11, color=MUTED)
        _add_text(slide, f"{random_error:.6f}", 8.82, 5.32, 2.4, 0.38, size=19, color=CORAL, bold=True)
        _add_text(slide, f"Screened Random · {random_error/nflow_error:.0f}× 较大", 10.63, 5.36, 1.55, 0.34, size=10.5, color=MUTED)
        _add_text(slide, "图像由 NovaTurbo 19D参数、组件装配和STL网格导出链生成；为展示内部组件，渲染时隐藏外壳。", 0.75, 6.35, 11.4, 0.3, size=10.5, color=MUTED)

    # 11 — Convergence
    slide = _blank_slide(prs); _title(slide, "10 · DOWNSTREAM SEARCH", "15代后优势没有被随机搜索抹平", 11)
    _add_picture(slide, convergence_chart, 0.55, 1.55, 8.2, 4.85)
    _panel(slide, 9.02, 1.7, 3.65, 4.5, PANEL)
    _add_text(slide, "相对 Screened Random\n15代后最佳误差改善", 9.36, 2.05, 2.8, 0.75, size=15, color=MUTED, bold=True)
    _add_text(slide, "25.6×", 9.33, 3.03, 2.5, 0.62, size=32, color=GREEN, bold=True)
    _add_text(slide, "cINN · 最低最终误差", 9.36, 3.65, 2.65, 0.3, size=13)
    _add_text(slide, "24.0×", 9.33, 4.28, 2.5, 0.55, size=29, color=CYAN, bold=True)
    _add_text(slide, "nflows · 最佳初始模型", 9.36, 4.84, 2.65, 0.3, size=13)
    _add_text(slide, "Generator：100%目标成功\nScreened Random：87.5%", 9.36, 5.43, 2.7, 0.55, size=13, color=WHITE)

    # 12 — Conclusion
    slide = _blank_slide(prs); _title(slide, "11 · CONCLUSION", "生成式模型的价值：把搜索起点变成目标附近的一组可用解", 12)
    conclusions = [
        ("精度", "nflows初始最佳误差比Screened Random低204.9倍", CYAN),
        ("评估效率", "首次命中约129次评估，对照约493次；约3.8倍减少", GREEN),
        ("解的质量", "nflows/cINN的40个初始个体全部命中三项性能容差", AMBER),
        ("稳定性", "ID、Interpolation、两个训练种子均保持优势", PURPLE),
    ]
    for i, (head, body, color) in enumerate(conclusions):
        y = 1.65 + i * 1.15
        _panel(slide, 0.75, y, 7.45, 0.88, PANEL)
        _add_text(slide, head, 1.02, y + 0.23, 1.05, 0.3, size=14, color=color, bold=True)
        _add_text(slide, body, 2.1, y + 0.19, 5.65, 0.4, size=15, bold=True)
    _panel(slide, 8.6, 1.65, 3.95, 4.32, PANEL_2)
    _add_text(slide, "当前模型选择", 8.95, 2.0, 2.8, 0.35, size=14, color=MUTED, bold=True)
    _add_text(slide, "nflows", 8.92, 2.55, 2.5, 0.58, size=31, color=CYAN, bold=True)
    _add_text(slide, "最佳 generation-0 初始化", 8.95, 3.14, 2.8, 0.35, size=14)
    _add_text(slide, "cINN", 8.92, 3.82, 2.5, 0.52, size=28, color=GREEN, bold=True)
    _add_text(slide, "15代后最低平均误差", 8.95, 4.35, 2.8, 0.35, size=14)
    _add_text(slide, "下一步", 8.95, 5.08, 1.0, 0.28, size=12, color=AMBER, bold=True)
    _add_text(slide, "面向更高成本 CFD/FEA 验证\n量化真实计算时间节省", 8.95, 5.43, 2.9, 0.55, size=13)
    _add_text(slide, "数据与代码：NovaTurbo · target_directed_active10d_gpu6", 0.75, 6.76, 7.2, 0.25, size=10, color=MUTED)

    # Common footer
    for index, slide in enumerate(prs.slides, start=1):
        if index == 1:
            continue
        _add_text(slide, "NovaTurbo · 目标导向反向设计", 0.58, 7.13, 3.2, 0.18, size=8.5, color=MUTED)
        _add_text(slide, f"{index:02d} / {len(prs.slides):02d}", 11.9, 7.13, 0.82, 0.18, size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)

    prs.save(OUTPUT_PPTX)
    return OUTPUT_PPTX


if __name__ == "__main__":
    path = build_presentation()
    print(path)
