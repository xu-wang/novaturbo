"""Multi-campaign WP19-21 expansion and deterministic streaming merge."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.physics.wp19_21_dataset import WP1921Error, run_wp19_21


class WP1921ExpansionError(ValueError):
    """Raised when an expansion campaign or merged dataset is invalid."""


def allocate_preview_budgets(total: int, campaign_count: int) -> list[int]:
    if total < 0 or campaign_count <= 0:
        raise WP1921ExpansionError("Invalid preview budget or campaign count")
    quotient, remainder = divmod(total, campaign_count)
    return [quotient + (index < remainder) for index in range(campaign_count)]


def _canonical_design_hash(design: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def merge_campaign_datasets(
    campaigns: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    expected_count: int,
) -> dict[str, Any]:
    """Stream campaigns into full, converged-regression, and feasibility sets."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    full_path = output / "dataset.jsonl"
    regression_path = output / "regression_converged.jsonl"
    feasibility_path = output / "feasibility_all.jsonl"
    temporary_paths = [path.with_suffix(path.suffix + ".tmp") for path in (full_path, regression_path, feasibility_path)]
    seen: set[str] = set()
    classification = Counter()
    total = converged = preview_requested = preview_available = 0
    try:
        with (
            temporary_paths[0].open("w", encoding="utf-8") as full_stream,
            temporary_paths[1].open("w", encoding="utf-8") as regression_stream,
            temporary_paths[2].open("w", encoding="utf-8") as feasibility_stream,
        ):
            for campaign in campaigns:
                campaign_id = str(campaign["campaign_id"])
                dataset_path = Path(campaign["dataset_jsonl"])
                with dataset_path.open(encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        design_hash = _canonical_design_hash(record["x_19"])
                        if design_hash in seen:
                            raise WP1921ExpansionError(
                                f"Duplicate x_19 across campaigns: {campaign_id}"
                            )
                        seen.add(design_hash)
                        local_id = int(record["sample_id"])
                        record["sample_id"] = total
                        record["campaign_id"] = campaign_id
                        record["campaign_sample_id"] = local_id
                        record["provenance"] = {
                            **record["provenance"],
                            "campaign_id": campaign_id,
                            "campaign_seed": int(campaign["seed"]),
                        }
                        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                        full_stream.write(encoded + "\n")
                        is_converged = bool(record["solver_status"]["converged"])
                        if is_converged:
                            regression_stream.write(encoded + "\n")
                            converged += 1
                        feasibility_stream.write(
                            json.dumps(
                                {
                                    "sample_id": total,
                                    "campaign_id": campaign_id,
                                    "campaign_sample_id": local_id,
                                    "x_19": record["x_19"],
                                    "converged": is_converged,
                                    "classification": record["solver_status"]["classification"],
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        classification[record["solver_status"]["classification"]] += 1
                        preview_requested += bool(record["mesh_preview_reference"]["requested"])
                        preview_available += bool(record["mesh_preview_reference"]["available"])
                        total += 1
        if total != expected_count:
            raise WP1921ExpansionError(
                f"Merged count {total} does not match expected {expected_count}"
            )
        for temporary, target in zip(temporary_paths, (full_path, regression_path, feasibility_path)):
            temporary.replace(target)
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise
    return {
        "sample_count": total,
        "unique_design_count": len(seen),
        "duplicate_design_count": total - len(seen),
        "converged_count": converged,
        "convergence_rate": converged / total,
        "classification_distribution": dict(sorted(classification.items())),
        "preview_requested_count": preview_requested,
        "preview_available_count": preview_available,
        "dataset_jsonl": str(full_path),
        "regression_jsonl": str(regression_path),
        "feasibility_jsonl": str(feasibility_path),
    }


def _render_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# WP19-21 98,304 点开发数据集扩容",
            "",
            f"> 状态：**{report['status']}**  ",
            "> 数据级别：`development_bootstrap`；正式用途允许：`false`",
            "",
            "## 结果",
            "",
            f"- 总样本：`{summary['sample_count']}`；唯一 19D：`{summary['unique_design_count']}`。",
            f"- 收敛：`{summary['converged_count']}`（`{summary['convergence_rate']:.2%}`）。",
            f"- Preview：`{summary['preview_available_count']}/{summary['preview_requested_count']}`。",
            "- 独立导出完整数据集、收敛回归集和全样本可行性分类集。",
            "",
            "当前 calibration 仍为 bootstrap，本数据集不得标记为正式标定数据。",
            "",
        ]
    )


def run_wp19_21_expansion(
    protocol_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_source = Path(protocol_path).resolve()
    protocol_bytes = protocol_source.read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    if protocol["dataset_tier"] != "development_bootstrap" or protocol["formal_use_allowed"] is not False:
        raise WP1921ExpansionError("Expansion must remain development_bootstrap")
    base = protocol["base_campaign"]
    base_manifest_path = root / base["manifest"]
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest["status"] != "COMPLETE" or base_manifest["summary"]["sample_count"] != int(base["sample_count"]):
        raise WP1921ExpansionError("Base campaign is incomplete")
    expansion_root = root / protocol["artifacts"]["runtime_root"]
    expansion_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "protocol_hash": sha256(protocol_bytes).hexdigest(),
        "base_resume_contract_hash": base_manifest["provenance"]["resume_contract_hash"],
        "expected_total_sample_count": int(protocol["expected_total_sample_count"]),
        "campaigns": protocol["expansion_campaigns"],
        "additional_preview_budget": int(protocol["additional_preview_budget"]),
    }
    contract_path = expansion_root / "expansion_contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise WP1921ExpansionError("Existing expansion contract differs")
    else:
        _write_json_atomic(contract_path, contract)

    campaigns = list(protocol["expansion_campaigns"])
    budgets = allocate_preview_budgets(int(protocol["additional_preview_budget"]), len(campaigns))
    campaign_reports = []
    base_protocol = root / protocol["base_protocol"]
    for campaign, preview_budget in zip(campaigns, budgets):
        campaign_root = expansion_root / "campaigns" / campaign["campaign_id"]
        manifest_path = campaign_root / "manifest.json"
        resume = campaign_root.exists()
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "COMPLETE":
                campaign_reports.append(existing)
                print(f"campaign {campaign['campaign_id']} already complete", flush=True)
                continue
        print(f"campaign {campaign['campaign_id']} starting", flush=True)
        report = run_wp19_21(
            base_protocol,
            root,
            output_root_override=campaign_root,
            sample_count_override=int(campaign["sample_count"]),
            preview_budget_override=preview_budget,
            workers_override=int(protocol["workers"]),
            shard_size_override=int(protocol["shard_size"]),
            seed_override=int(campaign["seed"]),
            campaign_id=str(campaign["campaign_id"]),
            resume=resume,
        )
        if report["status"] != "COMPLETE":
            raise WP1921ExpansionError(
                f"Campaign {campaign['campaign_id']} failed: {report['status']}"
            )
        campaign_reports.append(report)

    merge_inputs = [
        {
            "campaign_id": base["campaign_id"],
            "seed": base["seed"],
            "dataset_jsonl": root / base["dataset_jsonl"],
        }
    ]
    merge_inputs.extend(
        {
            "campaign_id": campaign["campaign_id"],
            "seed": campaign["seed"],
            "dataset_jsonl": expansion_root / "campaigns" / campaign["campaign_id"] / "dataset.jsonl",
        }
        for campaign in campaigns
    )
    merged_root = root / protocol["artifacts"]["merged_root"]
    summary = merge_campaign_datasets(
        merge_inputs,
        merged_root,
        int(protocol["expected_total_sample_count"]),
    )
    for key in ("dataset_jsonl", "regression_jsonl", "feasibility_jsonl"):
        summary[key] = str(Path(summary[key]).relative_to(root))
    expected_preview = int(base_manifest["summary"]["preview_requested_count"]) + int(protocol["additional_preview_budget"])
    status = "COMPLETE" if (
        summary["sample_count"] == int(protocol["expected_total_sample_count"])
        and summary["duplicate_design_count"] == 0
        and summary["preview_requested_count"] == expected_preview
        and summary["preview_available_count"] == expected_preview
    ) else "MERGE_AUDIT_FAILED"
    report = {
        "work_package": "WP19-21-expansion",
        "protocol_version": protocol["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_tier": protocol["dataset_tier"],
        "formal_use_allowed": False,
        "summary": summary,
        "campaigns": [
            {
                "campaign_id": base["campaign_id"],
                "seed": base["seed"],
                "summary": base_manifest["summary"],
            },
            *[
                {
                    "campaign_id": campaign["campaign_id"],
                    "seed": campaign["seed"],
                    "summary": campaign_report["summary"],
                }
                for campaign, campaign_report in zip(campaigns, campaign_reports)
            ],
        ],
        "provenance": {
            "expansion_protocol_hash": contract["protocol_hash"],
            "schema_hash": base_manifest["provenance"]["schema_hash"],
            "calibration_hash": base_manifest["provenance"]["calibration_hash"],
            "constraint_profile_hash": base_manifest["provenance"]["constraint_profile_hash"],
        },
    }
    _write_json_atomic(merged_root / "manifest.json", report)
    _write_json_atomic(root / protocol["artifacts"]["summary_json"], report)
    report_path = root / protocol["artifacts"]["report_markdown"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(report), encoding="utf-8")
    return report
