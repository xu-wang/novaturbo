#!/usr/bin/env python3
"""Regenerate the WP19-21 98,304-sample dataset with a configurable worker count.

Same campaign/merge protocol as ``scripts/run_wp19_21_dataset_expansion.py``,
but the number of physics workers is a CLI argument instead of a frozen 4. The
merged manifest keeps the protocol provenance hashes so downstream CFM training
still validates it.

Usage::

    python tools/v21/gen_dataset_fast.py --workers 12
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_surrogate import sha256_file
from src.physics.wp19_21_dataset import run_wp19_21
from src.physics.wp19_21_expansion import allocate_preview_budgets, merge_campaign_datasets


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "config/physics/wp19_21_dataset_expansion_98304_v1.yaml",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    base = protocol["base_campaign"]
    base_manifest_path = REPO_ROOT / base["manifest"]
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest["status"] != "COMPLETE":
        raise SystemExit("Base campaign is not complete; run the base script first")

    expansion_root = REPO_ROOT / protocol["artifacts"]["runtime_root"]
    campaigns = list(protocol["expansion_campaigns"])
    budgets = allocate_preview_budgets(int(protocol["additional_preview_budget"]), len(campaigns))
    reports = []
    for campaign, preview_budget in zip(campaigns, budgets):
        campaign_root = expansion_root / "campaigns" / campaign["campaign_id"]
        manifest_path = campaign_root / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "COMPLETE":
                print(f"campaign {campaign['campaign_id']} already complete", flush=True)
                reports.append(existing)
                continue
        print(f"campaign {campaign['campaign_id']} starting ({args.workers} workers)", flush=True)
        report = run_wp19_21(
            REPO_ROOT / protocol["base_protocol"],
            REPO_ROOT,
            output_root_override=campaign_root,
            sample_count_override=int(campaign["sample_count"]),
            preview_budget_override=preview_budget,
            workers_override=args.workers,
            shard_size_override=int(protocol["shard_size"]),
            seed_override=int(campaign["seed"]),
            campaign_id=str(campaign["campaign_id"]),
            resume=args.resume or campaign_root.exists(),
        )
        if report["status"] != "COMPLETE":
            raise SystemExit(f"campaign {campaign['campaign_id']} failed: {report['status']}")
        reports.append(report)

    merge_inputs = [
        {
            "campaign_id": base["campaign_id"],
            "seed": base["seed"],
            "dataset_jsonl": REPO_ROOT / base["dataset_jsonl"],
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
    merged_root = REPO_ROOT / protocol["artifacts"]["merged_root"]
    summary = merge_campaign_datasets(
        merge_inputs, merged_root, int(protocol["expected_total_sample_count"])
    )
    for key in ("dataset_jsonl", "regression_jsonl", "feasibility_jsonl"):
        summary[key] = str(Path(summary[key]).relative_to(REPO_ROOT))
    expected_preview = int(base_manifest["summary"]["preview_requested_count"]) + int(
        protocol["additional_preview_budget"]
    )
    status = "COMPLETE" if (
        summary["sample_count"] == int(protocol["expected_total_sample_count"])
        and summary["duplicate_design_count"] == 0
        and summary["preview_requested_count"] == expected_preview
        and summary["preview_available_count"] == expected_preview
    ) else "MERGE_AUDIT_FAILED"
    manifest = {
        "work_package": "WP19-21-expansion",
        "protocol_version": protocol["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_tier": "development_bootstrap",
        "formal_use_allowed": False,
        "summary": summary,
        "campaigns": [
            {"campaign_id": base["campaign_id"], "seed": base["seed"],
             "summary": base_manifest["summary"]},
            *[
                {"campaign_id": campaign["campaign_id"], "seed": campaign["seed"],
                 "summary": report["summary"]}
                for campaign, report in zip(campaigns, reports)
            ],
        ],
        "provenance": dict(base_manifest["provenance"]),
        "generation": {
            "generator": "tools/v21/gen_dataset_fast.py",
            "workers": args.workers,
            "protocol_hash": sha256_file(args.protocol),
        },
    }
    _write_json(merged_root / "manifest.json", manifest)
    print(json.dumps({"status": status, **summary}, sort_keys=True))
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
