#!/usr/bin/env python3
"""Exploratory C8/C128 replay of three previously viewed V3 C misses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_v3_c_window_comparison as comparison
from src.ai import geometry22_pipeline_v3_guided as p

SOURCE = ROOT / 'data/validation/v3_refinement_expanded_check_v1'
OUTPUT = ROOT / 'data/validation/v3_c_window_miss_appendix_v1'
CODE = ROOT / 'scripts/run_v3_c_window_miss_appendix.py'


def sources():
    previous = p.read(SOURCE / 'summary.json')
    ids = {row['target_id'] for row in previous['cases'] if not row['C_valid_hit']}
    p.require(len(ids) == 3, 'Previously viewed C-miss cohort changed')
    found = {}
    for target_file in (SOURCE / 'cases').glob('*/run/target.json'):
        target = p.read(target_file)
        if target['target_id'] in ids:
            found[target['target_id']] = target_file.parent
    p.require(set(found) == ids, 'Previous C-miss artifacts missing')
    return found


def freeze(output):
    p.require(not output.exists(), 'Refusing to overwrite appendix')
    cfg, _, _, _, _, _ = p.source(ROOT)
    selected = sources()
    files = {str(path.relative_to(ROOT)): p.sha(path)
             for run in selected.values() for path in
             (run / 'target.json', run / 'B/raw_candidates.json', run / 'B/candidates.json',
              run / 'B/route.json', run / 'C/evaluations.json')}
    lock = {'version': 'v3_c_window_miss_appendix_v1', 'scope': 'viewed_failures_exploratory_only',
            'runtime_target_ids': sorted(selected), 'source_files': files,
            'code_sha256': p.sha(CODE), 'comparison_code_sha256': p.sha(ROOT / comparison.CODE),
            'guided_code_sha256': p.sha(ROOT / 'src/ai/geometry22_pipeline_v3_guided.py'),
            'pipeline_config_sha256': p.sha(ROOT / p.CONFIG), 'policy': comparison.POLICY,
            'formal_use_allowed': False, 'deployment_allowed': False}
    output.mkdir(parents=True)
    p.write(output / 'protocol_lock.json', lock)
    print(json.dumps({'status': 'FROZEN', 'cases': len(selected), 'output': str(output)}), flush=True)


def check(output):
    lock = p.read(output / 'protocol_lock.json')
    p.require(lock['version'] == 'v3_c_window_miss_appendix_v1' and lock['policy'] == comparison.POLICY,
              'Appendix protocol changed')
    p.require(lock['code_sha256'] == p.sha(CODE)
              and lock['comparison_code_sha256'] == p.sha(ROOT / comparison.CODE)
              and lock['guided_code_sha256'] == p.sha(ROOT / 'src/ai/geometry22_pipeline_v3_guided.py')
              and lock['pipeline_config_sha256'] == p.sha(ROOT / p.CONFIG), 'Appendix code or config changed')
    p.require(lock['source_files'] == {name: p.sha(ROOT / name) for name in lock['source_files']},
              'Previous V3 artifacts changed')
    p.require(lock['runtime_target_ids'] == sorted(sources()), 'C-miss cohort changed')
    return lock


def execute(output, verify_only=False):
    lock = check(output)
    cfg, _, local, support, norm, checkpoint = p.source(ROOT)
    selected = sources()
    reports = []
    with p.run_lock(output):
        for target_id in lock['runtime_target_ids']:
            old = selected[target_id]
            target = p.read(old / 'target.json')
            case = output / 'cases' / target_id
            if not case.exists():
                p.require(not verify_only, 'Appendix case missing')
                raw = p.read(old / 'B/raw_candidates.json')
                new = comparison.rank(raw)
                p.require(p.read(old / 'B/candidates.json') == new[:24], 'Old B prefix changed')
                case.mkdir(parents=True)
                p.write(case / 'raw.json', raw)
                p.write(case / 'route.json', p.read(old / 'B/route.json'))
                p.write(case / 'selected.json', new)
            row = {'target_performance': {n: target[n] for n in p.OUTPUT_NAMES},
                   'runtime_target_id': target_id, 'source_center_id': 'previously_viewed'}
            report = (comparison.verify_case(case, target, cfg) if verify_only else
                      comparison.run_case(ROOT, case, row, cfg, local, support, norm, checkpoint))
            reports.append(report)
            print(json.dumps({'target_id': target_id, 'C8': report['arms']['C8']['hit'],
                              'C128': report['arms']['C128']['hit'],
                              'first_valid_hit_rank': report['first_valid_hit_rank']}), flush=True)
        summary = comparison.summarize(reports)
        summary['scope'] = 'viewed_failures_exploratory_only'
        if verify_only:
            p.require(p.read(output / 'summary.json') == summary, 'Appendix summary changed')
        else:
            p.write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'verify'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.output)
    else:
        print(json.dumps(execute(args.output, verify_only=args.command == 'verify'), indent=2))
