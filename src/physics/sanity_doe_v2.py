"""WP19-10 small-DOE sampling, failure analysis, and sensitivity audit."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc, spearmanr
import yaml

from src.physics.engine_match import EngineResult, Environment, solve_design_point
from src.physics.geometry_state import Geometry19
from src.physics.model_config import ModelConfig


class WP19DOEError(ValueError):
    """Raised when a WP19-10 protocol or generated artifact is invalid."""


@dataclass(frozen=True)
class WP19DOEResult:
    """In-memory summary and flat sample ledger for one deterministic DOE."""

    report: Mapping[str, Any]
    sample_records: tuple[Mapping[str, Any], ...]


def _load_yaml(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise WP19DOEError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise WP19DOEError(f"{label} must contain a mapping")
    return payload, sha256(raw).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WP19DOEError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise WP19DOEError(f"{name} must be finite")
    return converted


def _variables(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = schema.get("design_variables")
    if not isinstance(raw, list) or len(raw) != 19:
        raise WP19DOEError("Geometry19 schema must contain 19 variables")
    if tuple(item.get("name") for item in raw) != Geometry19.coordinate_names():
        raise WP19DOEError("Geometry19 schema coordinate order is not canonical")
    return raw


def sample_geometry19_designs(
    schema_path: str | Path,
    sample_count: int,
    seed: int,
) -> tuple[Geometry19, ...]:
    """Generate a reproducible mixed continuous/discrete Geometry19 LHS."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise WP19DOEError("sample_count must be an integer")
    if sample_count <= 0:
        raise WP19DOEError("sample_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise WP19DOEError("seed must be an integer")
    schema, _ = _load_yaml(Path(schema_path).resolve(), "design schema")
    variables = _variables(schema)
    unit_samples = qmc.LatinHypercube(d=19, seed=seed).random(sample_count)
    designs: list[Geometry19] = []
    for unit_row in unit_samples:
        values: dict[str, float | int] = {}
        for value, variable in zip(unit_row, variables):
            name = str(variable["name"])
            bounds = variable.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise WP19DOEError(f"{name} has invalid bounds")
            lower, upper = bounds
            if variable.get("dtype") == "int64":
                level_count = int(upper) - int(lower) + 1
                values[name] = int(lower) + min(
                    int(math.floor(float(value) * level_count)),
                    level_count - 1,
                )
            elif variable.get("dtype") == "float64":
                values[name] = float(lower) + float(value) * (
                    float(upper) - float(lower)
                )
            else:
                raise WP19DOEError(f"{name} has unsupported dtype")
        designs.append(Geometry19.from_mapping(values))
    return tuple(designs)


def _environment(protocol: Mapping[str, Any]) -> Environment:
    raw = protocol.get("environment")
    if not isinstance(raw, Mapping):
        raise WP19DOEError("DOE environment must be a mapping")
    return Environment(
        temperature_K=_finite_number(raw.get("temperature_K"), "temperature_K"),
        pressure_Pa=_finite_number(raw.get("pressure_Pa"), "pressure_Pa"),
        mach=_finite_number(raw.get("mach"), "mach"),
    )


def _result_record(
    sample_id: int,
    design: Geometry19,
    result: EngineResult,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_id": sample_id,
        **design.as_mapping(),
        "solver_converged": result.solver.converged,
        "failure_code": (
            result.solver.failure_code.value if result.solver.failure_code else ""
        ),
        "solver_evaluations": result.solver.evaluations,
        "solver_iterations": result.solver.iterations,
        "flow_residual": (
            result.solver.flow_residual
            if math.isfinite(result.solver.flow_residual)
            else None
        ),
        "thermodynamic_feasible": result.thermodynamic_feasible,
        "feasible": result.feasible,
        "mass_flow_air_kg_s": result.operating_point.get("mass_flow_air_kg_s"),
        "thrust_N": result.performance.get("thrust_N"),
        "tsfc_kg_N_s": result.performance.get("tsfc_kg_N_s"),
        "mass_kg": result.performance.get("mass_kg"),
    }
    if result.solver.converged:
        violations = {
            name: float(value)
            for name, value in result.constraints.items()
            if not math.isfinite(float(value)) or float(value) > 0.0
        }
        assembly_feasible = all(
            math.isfinite(float(value)) and float(value) <= 0.0
            for value in result.assembly_constraints.values()
        )
        record.update(
            {
                "max_constraint_residual": max(result.constraints.values()),
                "violated_constraint_count": len(violations),
                "violated_constraints": ";".join(sorted(violations)),
                "assembly_feasible": assembly_feasible,
            }
        )
    else:
        record.update(
            {
                "max_constraint_residual": None,
                "violated_constraint_count": None,
                "violated_constraints": "",
                "assembly_feasible": False,
            }
        )
    return record


def _distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size == 0:
        return None
    quantiles = np.quantile(finite, [0.05, 0.5, 0.95])
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _physics_channels(result: EngineResult) -> dict[str, float]:
    if not result.solver.converged:
        return {}
    channels = {
        f"performance.{name}": float(value)
        for name, value in result.performance.items()
    }
    channels.update(
        {
            f"constraint.{name}": float(value)
            for name, value in result.constraints.items()
        }
    )
    return channels


def _oat_alternatives(
    variable: Mapping[str, Any],
    fraction: float,
) -> tuple[float | int, ...]:
    lower, upper = variable["bounds"]
    reference = variable["reference_value"]
    if variable["dtype"] == "int64":
        candidates = (int(reference) + 1, int(reference) - 1)
        return tuple(value for value in candidates if lower <= value <= upper)
    delta = fraction * (float(upper) - float(lower))
    candidates = (float(reference) + delta, float(reference) - delta)
    return tuple(value for value in candidates if lower <= value <= upper)


def _oat_sensitivity(
    variables: Sequence[Mapping[str, Any]],
    reference: Geometry19,
    baseline: EngineResult,
    environment: Environment,
    config: ModelConfig,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    fraction = _finite_number(
        settings.get("oat_fraction_of_range"), "oat_fraction_of_range"
    )
    relative_tolerance = _finite_number(
        settings.get("numeric_change_relative_tolerance"),
        "numeric_change_relative_tolerance",
    )
    absolute_tolerance = _finite_number(
        settings.get("numeric_change_absolute_tolerance"),
        "numeric_change_absolute_tolerance",
    )
    if not 0.0 < fraction <= 0.5:
        raise WP19DOEError("OAT range fraction must be in (0, 0.5]")
    baseline_channels = _physics_channels(baseline)
    if not baseline_channels:
        raise WP19DOEError("Reference design must converge for OAT audit")
    audits: dict[str, Any] = {}
    reference_values = reference.as_mapping()
    for variable in variables:
        name = str(variable["name"])
        attempts: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for alternative in _oat_alternatives(variable, fraction):
            values = dict(reference_values)
            values[name] = alternative
            result = solve_design_point(
                Geometry19.from_mapping(values), environment, config
            )
            failure_code = (
                result.solver.failure_code.value
                if result.solver.failure_code
                else None
            )
            attempt = {
                "alternative_value": alternative,
                "converged": result.solver.converged,
                "failure_code": failure_code,
            }
            attempts.append(attempt)
            channels = _physics_channels(result)
            if not channels:
                continue
            changed: dict[str, float] = {}
            for channel in sorted(set(baseline_channels) & set(channels)):
                before = baseline_channels[channel]
                after = channels[channel]
                if not math.isclose(
                    before,
                    after,
                    rel_tol=relative_tolerance,
                    abs_tol=absolute_tolerance,
                ):
                    changed[channel] = after - before
            attempt["changed_channel_count"] = len(changed)
            if changed:
                selected = {
                    "alternative_value": alternative,
                    "changed_channels": changed,
                }
                break
        audits[name] = {
            "reference_value": reference_values[name],
            "attempts": attempts,
            "selected_alternative": (
                selected["alternative_value"] if selected else None
            ),
            "changed_channels": selected["changed_channels"] if selected else {},
            "dead_dimension": selected is None,
        }
    dead = [name for name, audit in audits.items() if audit["dead_dimension"]]
    return {
        "variables": audits,
        "dead_dimensions": dead,
        "passed": not dead,
    }


def _statistical_sensitivity(
    designs: Sequence[Geometry19],
    results: Sequence[EngineResult],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    minimum_pairs = int(settings.get("spearman_minimum_pairs", 20))
    top_count = int(settings.get("reported_top_channels", 5))
    if minimum_pairs < 3 or top_count <= 0:
        raise WP19DOEError("Invalid statistical-sensitivity settings")
    design_mappings = [design.as_mapping() for design in designs]
    channel_mappings = [_physics_channels(result) for result in results]
    all_channels = sorted(
        {channel for mapping in channel_mappings for channel in mapping}
    )
    report: dict[str, Any] = {}
    for name in Geometry19.coordinate_names():
        correlations: list[dict[str, Any]] = []
        for channel in all_channels:
            pairs = [
                (float(design[name]), float(channels[channel]))
                for design, channels in zip(design_mappings, channel_mappings)
                if channel in channels
                and math.isfinite(float(channels[channel]))
            ]
            if len(pairs) < minimum_pairs:
                continue
            x = np.asarray([pair[0] for pair in pairs])
            y = np.asarray([pair[1] for pair in pairs])
            if np.unique(x).size < 2 or np.unique(y).size < 2:
                continue
            correlation = float(spearmanr(x, y).statistic)
            if math.isfinite(correlation):
                correlations.append(
                    {
                        "channel": channel,
                        "spearman": correlation,
                        "pair_count": len(pairs),
                    }
                )
        correlations.sort(key=lambda item: abs(item["spearman"]), reverse=True)
        report[name] = {
            "maximum_absolute_spearman": (
                abs(correlations[0]["spearman"]) if correlations else None
            ),
            "top_channels": correlations[:top_count],
        }
    return report


def _summarize(
    records: Sequence[Mapping[str, Any]],
    results: Sequence[EngineResult],
) -> dict[str, Any]:
    sample_count = len(records)
    converged = [record for record in records if record["solver_converged"]]
    feasible = [record for record in converged if record["feasible"]]
    thermo_feasible = [
        record for record in converged if record["thermodynamic_feasible"]
    ]
    assembly_feasible = [
        record for record in converged if record["assembly_feasible"]
    ]
    failures: dict[str, int] = {}
    violations: dict[str, int] = {}
    for record in records:
        if not record["solver_converged"]:
            code = str(record["failure_code"] or "UNCLASSIFIED_FAILURE")
            failures[code] = failures.get(code, 0) + 1
        for name in str(record["violated_constraints"]).split(";"):
            if name:
                violations[name] = violations.get(name, 0) + 1
    performance = {
        name: _distribution(
            [float(record[name]) for record in converged if record[name] is not None]
        )
        for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")
    }
    return {
        "sample_count": sample_count,
        "converged_count": len(converged),
        "convergence_rate": len(converged) / sample_count,
        "hard_failure_count": sample_count - len(converged),
        "soft_infeasible_count": len(converged) - len(feasible),
        "feasible_count": len(feasible),
        "feasible_given_converged_rate": (
            len(feasible) / len(converged) if converged else 0.0
        ),
        "thermodynamic_feasible_given_converged_rate": (
            len(thermo_feasible) / len(converged) if converged else 0.0
        ),
        "assembly_feasible_given_converged_rate": (
            len(assembly_feasible) / len(converged) if converged else 0.0
        ),
        "multiple_operating_point_count": sum(
            result.solver.failure_code is not None
            and result.solver.failure_code.value == "MULTIPLE_OPERATING_POINTS"
            for result in results
        ),
        "failure_distribution": dict(sorted(failures.items())),
        "failure_rate_distribution": {
            name: count / sample_count for name, count in sorted(failures.items())
        },
        "constraint_violation_distribution": dict(
            sorted(violations.items(), key=lambda item: (-item[1], item[0]))
        ),
        "constraint_violation_rate_given_converged": {
            name: count / len(converged) if converged else 0.0
            for name, count in sorted(
                violations.items(), key=lambda item: (-item[1], item[0])
            )
        },
        "performance_distribution": performance,
    }


def run_wp19_10_doe(
    protocol_path: str | Path,
    repo_root: str | Path | None = None,
    sample_count_override: int | None = None,
) -> WP19DOEResult:
    """Execute WP19-10 and return the summary plus the flat sample ledger."""
    protocol_source = Path(protocol_path).resolve()
    protocol, protocol_hash = _load_yaml(protocol_source, "DOE protocol")
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else protocol_source.parents[2]
    )
    schema_path = root / str(protocol.get("design_schema", ""))
    calibration_path = root / str(protocol.get("calibration_pack", ""))
    schema, schema_hash = _load_yaml(schema_path, "design schema")
    variables = _variables(schema)
    config = ModelConfig.from_yaml(calibration_path)
    if protocol.get("physics_spec_version") != config.physics_spec_version:
        raise WP19DOEError("DOE and calibration physics versions differ")
    if protocol.get("sampler") != "latin_hypercube":
        raise WP19DOEError("Only the frozen latin_hypercube sampler is supported")
    configured_count = int(protocol.get("sample_count", 0))
    sample_count = (
        sample_count_override
        if sample_count_override is not None
        else configured_count
    )
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise WP19DOEError("Effective sample count must be an integer")
    if sample_count <= 0:
        raise WP19DOEError("Effective sample count must be positive")
    seed = int(protocol.get("seed"))
    environment = _environment(protocol)
    designs = sample_geometry19_designs(schema_path, sample_count, seed)
    results = tuple(
        solve_design_point(design, environment, config) for design in designs
    )
    records = tuple(
        _result_record(index, design, result)
        for index, (design, result) in enumerate(zip(designs, results))
    )
    sensitivity_settings = protocol.get("sensitivity")
    if not isinstance(sensitivity_settings, Mapping):
        raise WP19DOEError("DOE sensitivity settings must be a mapping")
    reference = Geometry19.from_mapping(
        {
            str(variable["name"]): variable["reference_value"]
            for variable in variables
        }
    )
    reference_result = solve_design_point(reference, environment, config)
    oat = _oat_sensitivity(
        variables,
        reference,
        reference_result,
        environment,
        config,
        sensitivity_settings,
    )
    statistical = _statistical_sensitivity(
        designs, results, sensitivity_settings
    )
    summary = _summarize(records, results)
    numerical_audit_passed = (
        summary["converged_count"] > 0 and oat["passed"]
    )
    calibration_actions_required = (
        not config.formal_dataset_generation_ready
        or summary["feasible_count"] == 0
    )
    if not numerical_audit_passed:
        status = "FAIL_NUMERICAL_AUDIT"
    elif calibration_actions_required:
        status = "PASS_WITH_CALIBRATION_ACTIONS"
    else:
        status = "PASS"
    report = {
        "protocol_version": protocol.get("protocol_version"),
        "status": status,
        "wp19_10_numerical_audit_passed": numerical_audit_passed,
        "wp19_11_entry_authorized": numerical_audit_passed,
        "calibration_actions_required": calibration_actions_required,
        "run_kind": (
            "frozen_sample_count"
            if sample_count == configured_count
            else "sample_count_override"
        ),
        "configured_sample_count": configured_count,
        "effective_sample_count": sample_count,
        "seed": seed,
        "environment": vars(environment),
        "formal_dataset_generation_ready": (
            config.formal_dataset_generation_ready
        ),
        "calibration_status": config.status,
        "summary": summary,
        "sensitivity_audit": {
            "oat_physical_path": oat,
            "statistical_spearman": statistical,
        },
        "provenance": {
            "physics_spec_version": config.physics_spec_version,
            "design_schema_version": config.design_schema_version,
            "calibration_pack_version": config.pack_version,
            "calibration_pack_hash": config.calibration_pack_hash,
            "calibration_source_file_hash": config.source_file_hash,
            "design_schema_file_hash": schema_hash,
            "doe_protocol_file_hash": protocol_hash,
        },
    }
    return WP19DOEResult(report=report, sample_records=records)


def render_wp19_10_markdown(report: Mapping[str, Any]) -> str:
    """Render the WP19-10 summary and sensitivity gate."""
    summary = report["summary"]
    lines = [
        "# WP19-10 小规模 DOE 报告",
        "",
        f"- 协议：`{report['protocol_version']}`",
        f"- 运行类型：`{report['run_kind']}`",
        f"- 样本数：{report['effective_sample_count']}",
        f"- 随机种子：{report['seed']}",
        f"- 结论：`{report['status']}`",
        "- 正式数据生成许可："
        f"`{str(report['formal_dataset_generation_ready']).lower()}`",
        "",
        "## 求解与可行性",
        "",
        f"- 收敛率：{summary['convergence_rate']:.6f}",
        "- 收敛后正式可行率："
        f"{summary['feasible_given_converged_rate']:.6f}",
        "- 收敛后热力可行率："
        f"{summary['thermodynamic_feasible_given_converged_rate']:.6f}",
        "- 收敛后装配可行率："
        f"{summary['assembly_feasible_given_converged_rate']:.6f}",
        f"- 多工作点数量：{summary['multiple_operating_point_count']}",
        "",
        "## Hard failure 分布",
        "",
        "| failure code | count | probability |",
        "|---|---:|---:|",
    ]
    failures = summary["failure_distribution"]
    if failures:
        lines.extend(
            "| `{name}` | {count} | {rate:.6f} |".format(
                name=name,
                count=count,
                rate=summary["failure_rate_distribution"][name],
            )
            for name, count in failures.items()
        )
    else:
        lines.append("| 无 | 0 | 0.000000 |")
    lines.extend(
        [
            "",
            "## 高频约束超限",
            "",
            "| constraint | count |",
            "|---|---:|",
        ]
    )
    violations = summary["constraint_violation_distribution"]
    for name, count in list(violations.items())[:15]:
        lines.append(f"| `{name}` | {count} |")
    oat = report["sensitivity_audit"]["oat_physical_path"]
    lines.extend(
        [
            "",
            "## 19D 物理通道审计",
            "",
            f"- 死维度数量：{len(oat['dead_dimensions'])}",
            "- 死维度："
            + (", ".join(oat["dead_dimensions"]) if oat["dead_dimensions"] else "无"),
            "",
            "| variable | changed channels | strongest statistical channel | |rho| |",
            "|---|---:|---|---:|",
        ]
    )
    statistical = report["sensitivity_audit"]["statistical_spearman"]
    for name, audit in oat["variables"].items():
        top = statistical[name]["top_channels"]
        strongest = top[0] if top else None
        lines.append(
            "| `{name}` | {changed} | {channel} | {rho} |".format(
                name=name,
                changed=len(audit["changed_channels"]),
                channel=(f"`{strongest['channel']}`" if strongest else "—"),
                rho=(f"{abs(strongest['spearman']):.6f}" if strongest else "—"),
            )
        )
    lines.extend(
        [
            "",
            "该报告只授权后续 calibration audit。标定包仍未冻结，",
            "不得将本轮样本作为正式大规模训练数据发布。",
            "",
        ]
    )
    return "\n".join(lines)


def write_wp19_10_artifacts(
    result: WP19DOEResult,
    report_json: str | Path,
    sample_ledger_csv: str | Path,
    report_markdown: str | Path,
) -> None:
    """Write strict JSON, flat CSV, and Markdown artifacts."""
    json_path = Path(report_json)
    csv_path = Path(sample_ledger_csv)
    markdown_path = Path(report_markdown)
    for path in (json_path, csv_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    if not result.sample_records:
        raise WP19DOEError("Cannot write an empty DOE ledger")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result.sample_records[0].keys())
        writer.writeheader()
        writer.writerows(result.sample_records)
    markdown_path.write_text(
        render_wp19_10_markdown(result.report), encoding="utf-8"
    )
