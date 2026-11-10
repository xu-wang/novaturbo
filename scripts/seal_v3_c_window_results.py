#!/usr/bin/env python3
"""Seal or verify completed V3 C-window experiment files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai import geometry22_pipeline_v3_guided as p

OUTPUTS = (
    ROOT / 'data/validation/v3_c_window_comparison_v1',
    ROOT / 'data/validation/v3_c_window_miss_appendix_v1',
)


def manifest(output):
    p.require(p.read(output / 'summary.json')['status'] == 'COMPLETE', 'Experiment incomplete')
    files = {str(path.relative_to(output)): p.sha(path) for path in sorted(output.rglob('*'))
             if path.is_file() and path.name not in ('.runner.lock', 'complete_manifest.json')}
    return {'version': 'v3_c_window_file_manifest_v1',
            'protocol_lock_sha256': files['protocol_lock.json'], 'files': files}


def process(output, verify):
    path = output / 'complete_manifest.json'
    value = manifest(output)
    if verify:
        p.require(p.read(path) == value, 'C-window file manifest changed')
    else:
        p.require(not path.exists(), 'Refusing to replace complete manifest')
        p.write(path, value)
    return {'output': str(output), 'files': len(value['files']), 'verified': verify}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('seal', 'verify'))
    args = parser.parse_args()
    for directory in OUTPUTS:
        print(process(directory, args.command == 'verify'))
