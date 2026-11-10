#!/usr/bin/env python3
"""Create a one-slide external-facing summary of the Geometry19 demo."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
DESIGN_ID = "0101c93ecfa4d70079a2015cb83362b47a8b7c1076c94004594deb2dc717b715"
MESH_ROOT = ROOT / "exports" / "geometry19" / DESIGN_ID / "preview_mesh"
OUT_DIR = ROOT / "reports" / "geometry19_external_demo"
RENDER_PATH = OUT_DIR / "geometry19_engine_render.png"
PPTX_PATH = OUT_DIR / "NovaTurbo_19D_外部展示_单页.pptx"
FLOW_IMAGE = Path("/home/wangjianyu/.codex/attachments/957210da-7966-4603-aeee-92439dbd8d13/codex-clipboard-16bdbb33-2b15-4483-9110-f260e8069361.png")
WORKSPACE_IMAGE = Path("/home/wangjianyu/.codex/attachments/4a54e5fa-aeb9-45f4-b037-6de3f587d4da/codex-clipboard-8a02d067-ec1a-4064-ae3c-6dae744c0e7b.png")

NAVY = "FFFFFF"
PANEL = "F7FAFC"
PANEL_2 = "FFFFFF"
WHITE = "172B3A"
MUTED = "637987"
CYAN = "27D6C5"
BLUE = "35A7FF"
ORANGE = "FFB24A"
GREEN = "57D68D"
RED = "FF6B6B"
FONT = "Noto Sans CJK SC"


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def render_engine() -> None:
    components = [
        ("inlet.stl", "#35A7FF"),
        ("compressor.stl", "#57D68D"),
        ("combustor.stl", "#FF8B5B"),
        ("combustor_lattice.stl", "#FFB24A"),
        ("transitions.stl", "#9C7CFF"),
        ("turbine.stl", "#FFD15C"),
        ("nozzle.stl", "#FF6B6B"),
        ("nozzle_lattice.stl", "#FF8A3D"),
        ("shaft.stl", "#CBD6DE"),
    ]
    fig = plt.figure(figsize=(10, 5), dpi=220)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0, 0, 0, 0))
    all_vertices = []
    for filename, color in components:
        path = MESH_ROOT / filename
        if not path.exists():
            continue
        mesh = trimesh.load_mesh(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
        else:
            meshes = [mesh]
        for part in meshes:
            vertices = np.asarray(part.vertices)
            faces = np.asarray(part.faces)
            if len(faces) > 18000:
                stride = int(np.ceil(len(faces) / 18000))
                faces = faces[::stride]
            poly = Poly3DCollection(vertices[faces], linewidths=0.02, alpha=0.96)
            poly.set_facecolor(color)
            poly.set_edgecolor((1, 1, 1, 0.06))
            ax.add_collection3d(poly)
            all_vertices.append(vertices)
    vertices = np.vstack(all_vertices)
    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.view_init(elev=19, azim=-54)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RENDER_PATH, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def add_text(slide, text, x, y, w, h, size=12, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, font=FONT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Inches(0.03)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_round_rect(slide, x, y, w, h, fill=PANEL, line=None, radius_shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    shape = slide.shapes.add_shape(radius_shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line or fill)
    return shape


def add_group(slide, title, lines, x, y, w, h, accent):
    add_round_rect(slide, x, y, w, h, fill=PANEL_2, line="D9E4EA")
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.055), Inches(h))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(accent); bar.line.fill.background()
    add_text(slide, title, x + 0.15, y + 0.08, w - 0.25, 0.24, 10.5, accent, True)
    box = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.36), Inches(w - 0.25), Inches(h - 0.42))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0; tf.margin_top = tf.margin_bottom = 0
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line; p.font.name = FONT; p.font.size = Pt(8.25); p.font.color.rgb = rgb(WHITE)
        p.space_after = Pt(1.5)


def add_metric(slide, label, value, unit, x, y, accent):
    add_round_rect(slide, x, y, 1.33, 0.84, fill="FFFFFF", line="D9E4EA")
    add_text(slide, label, x + 0.10, y + 0.08, 1.13, 0.18, 8.5, MUTED)
    add_text(slide, value, x + 0.10, y + 0.27, 1.13, 0.31, 17, accent, True)
    add_text(slide, unit, x + 0.10, y + 0.61, 1.13, 0.13, 7.2, MUTED)


def add_picture_crop(slide, path, x, y, w, h):
    """Fill a fixed frame while preserving image aspect ratio."""
    from PIL import Image
    with Image.open(path) as image:
        image_ratio = image.width / image.height
    frame_ratio = w / h
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    if image_ratio > frame_ratio:
        shown = frame_ratio / image_ratio
        pic.crop_left = pic.crop_right = (1 - shown) / 2
    else:
        shown = image_ratio / frame_ratio
        pic.crop_top = pic.crop_bottom = (1 - shown) / 2
    return pic


def build_pptx() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill; bg.solid(); bg.fore_color.rgb = rgb(NAVY)

    # Header
    add_text(slide, "NovaTurbo 19D 智能逆向设计", 0.48, 0.23, 7.6, 0.43, 24, WHITE, True)
    add_text(slide, "从目标性能输入，到 19D 方案生成、物理评估与交互可视化", 0.50, 0.68, 8.8, 0.23, 10.5, MUTED)
    add_round_rect(slide, 10.64, 0.30, 2.18, 0.46, fill="EAF9F7", line="B8E9E3")
    add_text(slide, "目标驱动生成设计", 10.64, 0.30, 2.18, 0.46, 11.5, CYAN, True, PP_ALIGN.CENTER)

    # Model input and achieved performance
    add_round_rect(slide, 0.48, 1.08, 3.12, 2.28, fill=PANEL, line="D9E4EA")
    add_text(slide, "生成网络输入", 0.70, 1.27, 2.20, 0.28, 14, WHITE, True)
    add_text(slide, "目标性能参数", 0.70, 1.58, 2.20, 0.20, 9, MUTED)
    target_rows = [
        ("目标推力", "63.143 N", BLUE),
        ("目标燃油消耗率", "6.526×10⁻⁵ kg/(N·s)", ORANGE),
        ("目标整机质量", "2.247 kg", GREEN),
    ]
    for i, (label, value, accent) in enumerate(target_rows):
        yy = 1.91 + i * 0.43
        add_text(slide, label, 0.72, yy, 1.22, 0.25, 8.5, MUTED)
        add_text(slide, value, 1.75, yy, 1.56, 0.25, 10.5, accent, True, PP_ALIGN.RIGHT)

    # Two supplied visual results
    add_round_rect(slide, 3.78, 1.08, 9.05, 2.28, fill=PANEL, line="D9E4EA")
    add_text(slide, "可视化结果", 3.99, 1.25, 2.0, 0.25, 13.5, WHITE, True)
    add_text(slide, "整机流动/工况展示", 4.02, 1.53, 2.45, 0.18, 8.2, MUTED)
    add_text(slide, "19D 设计与性能交互界面", 8.30, 1.53, 2.70, 0.18, 8.2, MUTED)
    add_picture_crop(slide, FLOW_IMAGE, 4.00, 1.77, 4.05, 1.38)
    add_picture_crop(slide, WORKSPACE_IMAGE, 8.27, 1.77, 4.34, 1.38)

    # More detailed process: seven stages
    add_text(slide, "完整流程", 0.50, 3.60, 1.5, 0.25, 13.5, WHITE, True)
    stages = [
        ("1", "提出需求", "定义推力/油耗/质量"),
        ("2", "目标标准化", "转换为模型条件"),
        ("3", "候选生成", "生成多组设计方案"),
        ("4", "19D 参数输出", "形成完整设计向量"),
        ("5", "物理评估", "回算性能与工况"),
        ("6", "筛选优化", "选择目标匹配方案"),
        ("7", "可视化交付", "生成三维与分析界面"),
    ]
    sx = 0.50
    box_w = 1.65
    gap = 0.15
    accents = [BLUE, BLUE, CYAN, CYAN, ORANGE, ORANGE, GREEN]
    for i, ((num, title, sub), accent) in enumerate(zip(stages, accents)):
        x = sx + i * (box_w + gap)
        add_round_rect(slide, x, 3.97, box_w, 0.84, fill="F7FAFC", line="D9E4EA")
        add_round_rect(slide, x + 0.09, 4.08, 0.30, 0.30, fill=accent, line=accent, radius_shape=MSO_SHAPE.OVAL)
        add_text(slide, num, x + 0.09, 4.08, 0.30, 0.30, 8.5, "FFFFFF", True, PP_ALIGN.CENTER)
        add_text(slide, title, x + 0.46, 4.05, 1.08, 0.25, 9.4, WHITE, True)
        add_text(slide, sub, x + 0.12, 4.42, 1.41, 0.20, 7.2, MUTED, False, PP_ALIGN.CENTER)

    # Generator output = complete 19D vector
    add_text(slide, "生成网络输出：完整 19D 设计向量", 0.50, 5.08, 5.4, 0.28, 13.5, WHITE, True)
    add_text(slide, "以下为本例生成参数（尺寸已换算为 mm）", 0.50, 5.38, 5.0, 0.19, 8.2, MUTED)
    add_group(slide, "压气机 · 6D", [
        "出口径 129.3 · 诱导轮尖径 78.9",
        "轮毂/叶尖比 0.371 · 出口宽 8.81",
        "出口叶角 57.6° · 叶片数 15",
    ], 0.50, 5.68, 2.38, 1.04, CYAN)
    add_group(slide, "燃烧室 · 5D", [
        "外径 108.7 · 内径 46.2 · 长度 78.8",
        "衬套厚度 1.03 · 喷嘴数量 12",
    ], 3.00, 5.68, 2.38, 1.04, ORANGE)
    add_group(slide, "涡轮/喷管 · 4D", [
        "涡轮尖径 106.9 · 轮毂/叶尖比 0.757",
        "叶片数 23 · 喷管喉径 73.6",
    ], 5.50, 5.68, 2.38, 1.04, BLUE)
    add_group(slide, "运行/结构 · 4D", [
        "进口温度 823.6 K · 转速 68,022 rpm",
        "晶格胞元 8.85 · 相对密度 0.284",
    ], 8.00, 5.68, 2.38, 1.04, GREEN)

    # Downstream evaluated results, explicitly separated from network output
    add_round_rect(slide, 10.52, 5.08, 2.30, 1.64, fill="F7FAFC", line="D9E4EA")
    add_text(slide, "物理评估结果", 10.72, 5.22, 1.90, 0.24, 11.5, WHITE, True)
    add_text(slide, "推力", 10.72, 5.60, 0.62, 0.20, 8, MUTED)
    add_text(slide, "63.40 N", 11.47, 5.58, 1.08, 0.22, 10.5, CYAN, True, PP_ALIGN.RIGHT)
    add_text(slide, "燃油消耗率", 10.72, 5.89, 0.86, 0.20, 8, MUTED)
    add_text(slide, "6.555×10⁻⁵", 11.54, 5.87, 1.01, 0.22, 9.6, ORANGE, True, PP_ALIGN.RIGHT)
    add_text(slide, "整机质量", 10.72, 6.18, 0.75, 0.20, 8, MUTED)
    add_text(slide, "2.257 kg", 11.47, 6.16, 1.08, 0.22, 10.5, GREEN, True, PP_ALIGN.RIGHT)
    add_text(slide, "目标匹配约 99.5%", 10.72, 6.48, 1.83, 0.16, 7.5, MUTED, False, PP_ALIGN.RIGHT)

    # Footer
    add_text(slide, "NOVA", 0.50, 6.95, 0.75, 0.20, 9.5, CYAN, True)
    add_text(slide, "TURBO", 1.18, 6.95, 0.85, 0.20, 9.5, WHITE, True)
    add_text(slide, "概念验证示例 · 用于展示方法流程与系统能力，非工程定型数据", 7.10, 6.94, 5.70, 0.18, 7.4, MUTED, False, PP_ALIGN.RIGHT)

    prs.save(PPTX_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_pptx()
    print(PPTX_PATH)
    print(RENDER_PATH)


if __name__ == "__main__":
    main()
