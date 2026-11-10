"""Cross-version feasibility migration audit for V2.1 to V2.2-C4."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.constraint_profiles import assess_constraint_profile
from src.physics.engine_match import Environment, Geometry19
from src.physics.flow_scan_diagnostics import diagnose_v2_2_c4_flow_scan


DEFAULT_SOURCE = Path(
    "data/generated/wp19_21_dataset_expansion_98304_v1/"
    "campaigns/seed_20260827/dataset.jsonl"
)
INCREMENTAL_PROFILES = (
    "v2_2_hecc_shadow",
    "v2_2_comp_turb_shadow",
    "v2_2_comp_turb_inlet_shadow",
    "v2_2_comp_turb_inlet_comb_shadow",
    "v2_2_full_component_shadow",
    "v2_2_full_variable_thermo_shadow",
)
SOFTWARE_PIPELINE_PROFILE = "v2_2_development"


class V22C5MigrationAuditError(ValueError):
    """Raised when a migration audit input is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_rows(path: Path, sample_limit: int) -> list[dict[str, Any]]:
    if sample_limit <= 0:
        raise V22C5MigrationAuditError("sample_limit must be positive")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("solver_status", {}).get("converged") is True:
                rows.append(row)
                if len(rows) == sample_limit:
                    break
    if len(rows) != sample_limit:
        raise V22C5MigrationAuditError(
            f"Requested {sample_limit} V2.1-converged rows, found {len(rows)}"
        )
    return rows


def run_v2_2_c5_migration_audit(
    repository_root: str | Path,
    *,
    source_dataset: str | Path | None = None,
    sample_limit: int = 256,
) -> dict[str, Any]:
    """Re-evaluate stored V2.1-converged designs with the C4 backend."""

    root = Path(repository_root).resolve()
    source = Path(source_dataset) if source_dataset is not None else root / DEFAULT_SOURCE
    source = source.resolve()
    rows = _source_rows(source, sample_limit)
    contexts = {
        name: build_pipeline_context(root, pipeline_profile=name)
        for name in INCREMENTAL_PROFILES
    }
    context = contexts[INCREMENTAL_PROFILES[-1]]
    software_context = build_pipeline_context(
        root, pipeline_profile=SOFTWARE_PIPELINE_PROFILE
    )
    software_counts: Counter[str] = Counter()
    software_failures: Counter[str] = Counter()
    incremental_counts = {name: Counter() for name in INCREMENTAL_PROFILES}
    incremental_failures = {name: Counter() for name in INCREMENTAL_PROFILES}
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    positive_constraints: Counter[str] = Counter()
    maximum_residuals: dict[str, float] = {}
    component_out_of_domain: Counter[str] = Counter()
    scan_failure_components: Counter[str] = Counter()
    scan_failure_codes: Counter[str] = Counter()
    scan_failure_reasons: Counter[str] = Counter()
    scan_design_classification: Counter[str] = Counter()
    compressor_query_coverage: Counter[str] = Counter()
    failed_sample_diagnostics = []
    converged_samples = []
    environment = Environment()

    for row in rows:
        design = Geometry19.from_mapping(row["x_19"])
        profile_results = {}
        for profile_name, profile_context in contexts.items():
            candidate = profile_context.physics_runtime.evaluate(design, environment)
            profile_results[profile_name] = candidate
            profile_counts = incremental_counts[profile_name]
            profile_counts["evaluated"] += 1
            profile_counts["converged"] += int(candidate.solver.converged)
            profile_counts["single_root"] += int(
                candidate.operating_point.get("flow_root_count") == 1
            )
            if not candidate.solver.converged:
                code = candidate.solver.failure_code
                incremental_failures[profile_name][
                    code.value if code is not None else "UNKNOWN"
                ] += 1
        result = profile_results[INCREMENTAL_PROFILES[-1]]
        software_result = software_context.physics_runtime.evaluate(design, environment)
        software_counts["evaluated"] += 1
        software_counts["converged"] += int(software_result.solver.converged)
        software_counts["single_root"] += int(
            software_result.operating_point.get("flow_root_count") == 1
        )
        if software_result.solver.converged:
            software_assessment = assess_constraint_profile(
                software_result,
                software_context.physics_runtime.constraint_profile,
                conservation_tolerances=(
                    software_context.physics_runtime.conservation_tolerances
                ),
            )
            software_counts["numerically_valid"] += int(
                software_assessment.numerically_valid
            )
            software_counts["physics_domain_valid"] += int(
                software_assessment.physics_domain_valid
            )
            software_counts["accepted"] += int(software_assessment.accepted)
            software_counts["thermodynamic_feasible"] += int(
                software_result.thermodynamic_feasible
            )
            software_counts["engine_feasible"] += int(software_result.feasible)
        else:
            code = software_result.solver.failure_code
            software_failures[
                code.value if code is not None else "UNKNOWN"
            ] += 1
        counts["evaluated"] += 1
        if not result.solver.converged:
            code = result.solver.failure_code
            failures[code.value if code is not None else "UNKNOWN"] += 1
            diagnostic = diagnose_v2_2_c4_flow_scan(context, design, environment)
            compressor_map = context.physics_runtime.component_maps["compressor"]
            speed_axis = compressor_map.coordinates["corrected_speed_ratio"]
            flow_axis = compressor_map.coordinates["flow_coefficient_ratio"]
            envelope = diagnostic["compressor_query_envelope"]
            speed = envelope["corrected_speed_ratio"]["minimum"]
            if not float(speed_axis[0]) <= speed <= float(speed_axis[-1]):
                compressor_query_coverage["speed_outside_raw_map"] += 1
            speed_span = float(speed_axis[-1] - speed_axis[0])
            speed_distance = max(
                float(speed_axis[0]) - speed,
                speed - float(speed_axis[-1]),
                0.0,
            ) / speed_span
            if speed_distance > compressor_map.diagnostic_extrapolation_limit:
                compressor_query_coverage["speed_outside_diagnostic_limit"] += 1
            flow_min = envelope["flow_coefficient_ratio"]["minimum"]
            flow_max = envelope["flow_coefficient_ratio"]["maximum"]
            if flow_max < float(flow_axis[0]) or flow_min > float(flow_axis[-1]):
                compressor_query_coverage["flow_scan_has_no_raw_map_overlap"] += 1
            if (
                envelope["domain_distance"]["minimum"]
                <= compressor_map.diagnostic_extrapolation_limit
            ):
                compressor_query_coverage[
                    "has_query_within_diagnostic_domain"
                ] += 1
            for name, count in diagnostic["first_failure_component_counts"].items():
                scan_failure_components[name] += count
            for name, count in diagnostic["failure_code_counts"].items():
                scan_failure_codes[name] += count
            for name, count in diagnostic["failure_reason_counts"].items():
                scan_failure_reasons[name] += count
            if diagnostic["valid_trial_count"] == 0:
                classification = "no_valid_component_chain_trials"
            elif diagnostic["sign_change_bracket_count"] > 0:
                classification = "sign_change_without_admitted_root"
            elif diagnostic["negative_residual_count"] == 0:
                classification = "all_valid_residuals_positive"
            elif diagnostic["positive_residual_count"] == 0:
                classification = "all_valid_residuals_negative"
            else:
                classification = "mixed_residuals_without_contiguous_bracket"
            scan_design_classification[classification] += 1
            failed_sample_diagnostics.append(
                {
                    "sample_id": row.get("sample_id"),
                    "classification": classification,
                    **{
                        key: value
                        for key, value in diagnostic.items()
                        if key != "valid_trials"
                    },
                }
            )
            continue
        counts["converged"] += 1
        counts["single_root"] += int(
            result.operating_point.get("flow_root_count") == 1
        )
        counts["thermodynamic_feasible"] += int(result.thermodynamic_feasible)
        counts["engine_feasible"] += int(result.feasible)
        assessment = assess_constraint_profile(
            result,
            context.physics_runtime.constraint_profile,
            conservation_tolerances=context.physics_runtime.conservation_tolerances,
        )
        counts["numerically_valid"] += int(assessment.numerically_valid)
        counts["physics_domain_valid"] += int(assessment.physics_domain_valid)
        map_status = {
            name: bool(component["map_in_domain"])
            for name, component in result.components.items()
            if "map_in_domain" in component
        }
        counts["all_maps_in_domain"] += int(
            bool(map_status) and all(map_status.values())
        )
        for name, in_domain in map_status.items():
            if not in_domain:
                component_out_of_domain[name] += 1
        positives = {
            name: float(value)
            for name, value in result.thermodynamic_constraints.items()
            if float(value) > 0.0
        }
        for name, value in positives.items():
            positive_constraints[name] += 1
            maximum_residuals[name] = max(maximum_residuals.get(name, 0.0), value)
        converged_samples.append(
            {
                "sample_id": row.get("sample_id"),
                "map_in_domain": map_status,
                "positive_thermodynamic_constraints": positives,
                "constraint_assessment": assessment.as_mapping(),
                "conservation": asdict(result.conservation),
            }
        )

    compressor_map = context.physics_runtime.component_maps["compressor"]
    compressor_coordinate_bounds = {
        name: [float(axis[0]), float(axis[-1])]
        for name, axis in compressor_map.coordinates.items()
    }
    audit_code_hash = sha256(
        json.dumps(
            {
                "flow_scan_diagnostics": _sha256_file(
                    root / "src/physics/flow_scan_diagnostics.py"
                ),
                "migration_audit": _sha256_file(Path(__file__).resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "audit_version": "geometry19_v2_2_c5_migration_audit_v1",
        "audit_code_hash": audit_code_hash,
        "source": {
            "dataset": str(source),
            "sha256": _sha256_file(source),
            "selection": "first_n_rows_with_stored_v2_1_solver_converged_true",
            "sample_limit": sample_limit,
        },
        "enhanced_profile": context.profile.name,
        "context_hash": context.context_hash,
        "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
        "counts": dict(sorted(counts.items())),
        "failure_codes": dict(sorted(failures.items())),
        "incremental_profile_migration": {
            name: {
                "counts": dict(sorted(incremental_counts[name].items())),
                "failure_codes": dict(sorted(incremental_failures[name].items())),
            }
            for name in INCREMENTAL_PROFILES
        },
        "software_pipeline_migration": {
            "profile": SOFTWARE_PIPELINE_PROFILE,
            "counts": dict(sorted(software_counts.items())),
            "failure_codes": dict(sorted(software_failures.items())),
        },
        "failed_flow_scan_summary": {
            "design_classification_counts": dict(
                sorted(scan_design_classification.items())
            ),
            "first_failure_component_trial_counts": dict(
                sorted(scan_failure_components.items())
            ),
            "failure_code_trial_counts": dict(sorted(scan_failure_codes.items())),
            "failure_reason_trial_counts": dict(
                sorted(scan_failure_reasons.items())
            ),
            "compressor_query_coverage": {
                "map_coordinate_bounds": compressor_coordinate_bounds,
                "diagnostic_extrapolation_limit": (
                    compressor_map.diagnostic_extrapolation_limit
                ),
                "failed_design_counts": dict(sorted(compressor_query_coverage.items())),
            },
        },
        "failed_sample_diagnostics": failed_sample_diagnostics,
        "component_out_of_domain_counts": dict(
            sorted(component_out_of_domain.items())
        ),
        "positive_thermodynamic_constraints": [
            {
                "name": name,
                "count": count,
                "maximum_residual": maximum_residuals[name],
            }
            for name, count in positive_constraints.most_common()
        ],
        "converged_samples": converged_samples,
        "training_ready": bool(
            counts["physics_domain_valid"] > 0
            and counts["thermodynamic_feasible"] > 0
        ),
    }


def render_v2_2_c5_migration_report(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# V2.1 → V2.2-C4 可行域迁移审计",
        "",
        f"- 样本数：{report['source']['sample_limit']}",
        f"- C4 收敛：{counts.get('converged', 0)}",
        f"- physics-domain valid：{counts.get('physics_domain_valid', 0)}",
        f"- 热力可行：{counts.get('thermodynamic_feasible', 0)}",
        f"- 训练就绪：**{str(report['training_ready']).lower()}**",
        "",
        "## 失败分类",
        "",
    ]
    lines.extend(
        f"- `{name}`：{count}" for name, count in report["failure_codes"].items()
    )
    lines.extend(["", "## 增量 profile 迁移", ""])
    for name, profile in report["incremental_profile_migration"].items():
        profile_counts = profile["counts"]
        lines.append(
            f"- `{name}`：收敛 {profile_counts.get('converged', 0)}/"
            f"{profile_counts.get('evaluated', 0)}"
        )
    software = report["software_pipeline_migration"]
    software_counts = software["counts"]
    lines.extend(["", "## V2.2 software pipeline profile", ""])
    lines.append(
        f"- `{software['profile']}`：收敛 "
        f"{software_counts.get('converged', 0)}/"
        f"{software_counts.get('evaluated', 0)}，physics-domain valid "
        f"{software_counts.get('physics_domain_valid', 0)}，"
        f"accepted {software_counts.get('accepted', 0)}"
    )
    lines.extend(["", "## C4 失败根扫描分类", ""])
    lines.extend(
        f"- `{name}`：{count}"
        for name, count in report["failed_flow_scan_summary"][
            "design_classification_counts"
        ].items()
    )
    lines.extend(["", "HECC 查询覆盖（按 C4 失败设计计数）：", ""])
    lines.extend(
        f"- `{name}`：{count}"
        for name, count in report["failed_flow_scan_summary"][
            "compressor_query_coverage"
        ]["failed_design_counts"].items()
    )
    lines.extend(["", "细分失败原因（按质量流量探针计数）：", ""])
    lines.extend(
        f"- `{name}`：{count}"
        for name, count in report["failed_flow_scan_summary"][
            "failure_reason_trial_counts"
        ].items()
    )
    lines.extend(["", "首个部件失败（按质量流量探针计数）：", ""])
    lines.extend(
        f"- `{name}`：{count}"
        for name, count in report["failed_flow_scan_summary"][
            "first_failure_component_trial_counts"
        ].items()
    )
    lines.extend(["", "## 收敛样本的正热力残差", ""])
    lines.extend(
        f"- `{item['name']}`：{item['count']}，最大值 {item['maximum_residual']:.6g}"
        for item in report["positive_thermodynamic_constraints"]
    )
    return "\n".join(lines) + "\n"


def write_v2_2_c5_migration_artifacts(
    report: dict[str, Any], output_root: str | Path
) -> None:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "migration_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "Migration_Audit.md").write_text(
        render_v2_2_c5_migration_report(report), encoding="utf-8"
    )
