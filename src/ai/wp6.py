"""WP6 G2 validation matrix built on the single-run inverse-design CLI."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from .baseline import sha256_file, write_json, write_yaml
from .inverse_design import load_inverse_design_config, run_inverse_design


REPRODUCIBILITY_ARTIFACTS = (
    "generator/raw_candidates.csv",
    "round_trip/ranked_candidates.csv",
    "population/initial_genes.csv",
    "population/population_provenance.csv",
    "optimization/pareto.csv",
    "optimization/history.csv",
    "optimization/evaluation_ledger.csv",
)


def load_wp6_validation_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "target_config",
        "run_configs",
        "reproducibility_run_config",
        "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP6 validation config is missing keys: {sorted(missing)}")
    if not config["run_configs"]:
        raise ValueError("WP6 validation requires at least one run config")
    return config


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _run_summary(run_dir: Path, run_config: Dict[str, Any]) -> Dict[str, Any]:
    metrics = json.loads(
        (run_dir / "summary" / "metrics.json").read_text(encoding="utf-8")
    )
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
    return {
        "run_id": run_config["run_id"],
        "mode": run_config["mode"],
        "model_id": run_config["model"].get("model_id"),
        "status": manifest["status"],
        "generated_count": metrics["population"]["generated_count"],
        "random_fill_count": metrics["population"]["random_fill_count"],
        "prefilter_0d_evaluations": metrics["cost"]["prefilter_0d_evaluations"],
        "nsga2_0d_evaluations": metrics["cost"]["nsga2_0d_evaluations"],
        "total_0d_evaluations": metrics["cost"]["total_0d_evaluations"],
        "pareto_size": metrics["optimization"]["pareto_size"],
        "counts_consistent": metrics["ledger_checks"]["counts_consistent"],
        "all_stages_completed": all(
            value["status"] == "completed" for value in manifest["stages"].values()
        ),
        "provenance_complete": bool(
            manifest.get("provenance")
            and (run_dir / "environment.json").is_file()
            and (run_dir / "git_state.json").is_file()
        ),
    }


def _write_report(
    path: Path,
    gate: Dict[str, Any],
    summary: pd.DataFrame,
    reproducibility: Dict[str, Any],
) -> None:
    lines = [
        "# WP6 G2 Pipeline Integration Gate",
        "",
        f"- G2: {'PASS' if gate['passed'] else 'FAIL'}",
        "- Generator superiority required: no",
        f"- Reproducibility: {'PASS' if reproducibility['passed'] else 'FAIL'}",
        "",
        "## Runs",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"- {row['run_id']}: mode={row['mode']}, model={row['model_id']}, "
            f"prefilter={row['prefilter_0d_evaluations']}, "
            f"nsga2={row['nsga2_0d_evaluations']}, "
            f"total={row['total_0d_evaluations']}, pareto={row['pareto_size']}"
        )
    lines.extend(
        [
            "",
            "G2 verifies functionality, reproducibility, cost separation, and "
            "provenance across the complete pipeline. It is not a Generator-vs-Random "
            "performance gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_wp6_validation(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_wp6_validation_config(config_path)
    target_path = _resolve(repo_root, config["target_config"])
    run_paths = [_resolve(repo_root, value) for value in config["run_configs"]]
    reproducibility_path = _resolve(
        repo_root, config["reproducibility_run_config"]
    )
    for path in (target_path, reproducibility_path, *run_paths):
        if not path.is_file():
            raise FileNotFoundError(f"WP6 validation input is missing: {path}")
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, config["output"]["directory"])
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"WP6 validation output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "runs").mkdir()
    (output_dir / "reproducibility").mkdir()
    shutil.copy2(config_path, output_dir / "validation_config.yaml")
    started = dt.datetime.now(dt.timezone.utc)

    records: List[Dict[str, Any]] = []
    run_configs = []
    for run_path in run_paths:
        run_config = load_inverse_design_config(run_path)
        run_configs.append(run_config)
        run_dir = run_inverse_design(
            target_path,
            run_path,
            repo_root=repo_root,
            output_dir_override=output_dir / "runs" / str(run_config["run_id"]),
        )
        records.append(_run_summary(run_dir, run_config))

    reproducibility_config = load_inverse_design_config(reproducibility_path)
    reference_dir = output_dir / "runs" / str(reproducibility_config["run_id"])
    if not reference_dir.is_dir():
        raise ValueError("reproducibility_run_config must also appear in run_configs")
    repeat_dir = run_inverse_design(
        target_path,
        reproducibility_path,
        repo_root=repo_root,
        output_dir_override=(
            output_dir / "reproducibility" / str(reproducibility_config["run_id"])
        ),
    )
    reproducibility_checks = {}
    for relative in REPRODUCIBILITY_ARTIFACTS:
        reference_hash = sha256_file(reference_dir / relative)
        repeat_hash = sha256_file(repeat_dir / relative)
        reproducibility_checks[relative] = {
            "reference_sha256": reference_hash,
            "repeat_sha256": repeat_hash,
            "match": reference_hash == repeat_hash,
        }
    reproducibility = {
        "run_id": reproducibility_config["run_id"],
        "passed": all(value["match"] for value in reproducibility_checks.values()),
        "checks": reproducibility_checks,
    }
    write_json(output_dir / "reproducibility.json", reproducibility)

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    modes = set(summary["mode"].astype(str))
    models = set(summary["model_id"].dropna().astype(str))
    expected_models = {"cvae", "nflows", "cinn", "cfm"}
    gate = {
        "gate": "G2 Pipeline Integration Gate",
        "passed": bool(
            set(["random", "generator", "hybrid"]).issubset(modes)
            and expected_models.issubset(models)
            and (summary["status"] == "completed").all()
            and summary["counts_consistent"].all()
            and summary["all_stages_completed"].all()
            and summary["provenance_complete"].all()
            and reproducibility["passed"]
        ),
        "generator_superiority_required": False,
        "required_modes_present": set(["random", "generator", "hybrid"]).issubset(modes),
        "all_models_present": expected_models.issubset(models),
        "all_runs_completed": bool((summary["status"] == "completed").all()),
        "all_cost_ledgers_consistent": bool(summary["counts_consistent"].all()),
        "all_provenance_complete": bool(summary["provenance_complete"].all()),
        "reproducible": reproducibility["passed"],
    }
    write_json(output_dir / "functional_gate.json", gate)
    _write_report(output_dir / "report.md", gate, summary, reproducibility)
    finished = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "wp": "WP6",
        "version": config["version"],
        "status": "completed" if gate["passed"] else "functional_gate_failed",
        "functional_gate_passed": gate["passed"],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "target_config": str(target_path),
        "target_config_sha256": sha256_file(target_path),
        "runs": records,
    }
    write_yaml(output_dir / "wp6_manifest.yaml", manifest)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the WP6 G2 validation matrix")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/inverse_design/wp6_validation.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_wp6_validation(
        args.config, output_dir_override=args.output_dir
    )
    manifest = yaml.safe_load((output / "wp6_manifest.yaml").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(output),
                "functional_gate_passed": manifest["functional_gate_passed"],
            }
        )
    )
    return 0 if manifest["functional_gate_passed"] else 1
