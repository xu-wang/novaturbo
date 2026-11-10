#!/usr/bin/env python3
"""Measure whether Diag-CFM intrinsic scores predict real NovaTurbo 0D error."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def analyze(evaluation_dirs: list[Path], output_csv: Path) -> pd.DataFrame:
    rows = []
    for evaluation_dir in evaluation_dirs:
        ledger = pd.read_csv(evaluation_dir / "evaluation_ledger.csv")
        raw_paths = sorted(
            (evaluation_dir / "artifacts" / "round_trip").glob(
                "*/raw_candidates.csv"
            )
        )
        raw = pd.concat((pd.read_csv(path) for path in raw_paths), ignore_index=True)
        diagnostic_columns = [
            name
            for name in ("intrinsic_zero_deviation", "intrinsic_self_consistency")
            if name in raw.columns
        ]
        merged = ledger.merge(
            raw[["model", "target_id", "candidate_id", *diagnostic_columns]],
            on=["model", "target_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        merged = merged[merged["forward_valid"].astype(bool)].copy()
        for model, model_frame in merged.groupby("model", sort=False):
            if not str(model).startswith("diag_cfm"):
                continue
            for score_name in diagnostic_columns:
                scored = model_frame.dropna(subset=[score_name, "total_error"])
                if scored.empty:
                    continue
                selected_frames = []
                for _, target_frame in scored.groupby("target_id", sort=False):
                    keep = max(1, len(target_frame) // 2)
                    selected_frames.append(target_frame.nsmallest(keep, score_name))
                selected = pd.concat(selected_frames, ignore_index=True)
                rows.append(
                    {
                        "evaluation": evaluation_dir.name,
                        "model": model,
                        "score": score_name,
                        "candidate_count": len(scored),
                        "spearman_score_vs_0d_error": float(
                            scored[score_name].corr(scored["total_error"], method="spearman")
                        ),
                        "all_hit_rate": float(scored["within_tolerance"].mean()),
                        "lowest_half_score_hit_rate": float(
                            selected["within_tolerance"].mean()
                        ),
                        "all_median_error": float(scored["total_error"].median()),
                        "lowest_half_score_median_error": float(
                            selected["total_error"].median()
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    frame = analyze(args.evaluation_dir, args.output_csv)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
