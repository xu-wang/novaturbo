#!/usr/bin/env python3
"""Verify a completed pipeline artifact without modifying or resuming it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_pipeline_artifact import (
    PipelineArtifactError, verify_completed_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_completed_artifact(args.output_root, REPO_ROOT)
    except PipelineArtifactError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
