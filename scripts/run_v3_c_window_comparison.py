#!/usr/bin/env python3
"""Paired V3 C8/C128 development comparison on one shared B draw per target."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai import geometry22_pipeline_v3_guided as p
from src.ai.geometry19_pipeline_v3 import _load_target_v3

REGISTRY = 'data/validation/v3_22d_extended_gate_l6f_gpu6_v1/target_registry.jsonl'
PREVIOUS_12 = 'data/validation/v3_refinement_difficult_check_v1/targets.json'
PREVIOUS_98 = 'data/validation/v3_refinement_expanded_check_v1/targets.json'
CODE = 'scripts/run_v3_c_window_comparison.py'
OUTPUT = 'data/validation/v3_c_window_comparison_v1'
POLICY = {
    'version': 'v3_c_window_comparison_v1', 'selected_candidate_cap': 128,
    'raw_slots': 256, 'windows': [8, 128], 'target_count': 110,
    'selection': '55 tuning centers; inside source fit box and P95 ball; distance ratio ranks 3 and 4',
    'scope': 'development_paired_diagnostic_not_independent_validation',
    'physical_gpu_index': 6, 'formal_use_allowed': False, 'deployment_allowed': False,
}


def choose_targets(rows, support, norm, excluded):
    by_center = {s['center_id']: s for s in support}
    grouped = {}
    for row in rows:
        if row['internal_split'] != 'tuning':
            continue
        center = by_center[row['source_center_id']]
        values = np.log([row['target_performance'][n] for n in p.OUTPUT_NAMES])
        normalized = (values - np.asarray(norm['log_mean'])) / np.asarray(norm['log_std'])
        radius = center['target_radius_p95_normalized_log']
        distance = float(np.linalg.norm(normalized - center['target_centroid_normalized_log']))
        if not (np.all(values >= center['target_log_min']) and np.all(values <= center['target_log_max'])
                and radius > 0 and distance <= radius):
            continue
        grouped.setdefault(row['source_center_id'], []).append({**row, 'source_support_distance_ratio': distance / radius})
    p.require(len(grouped) == 55 and all(len(v) >= 4 for v in grouped.values()), 'Insufficient eligible targets')
    key = lambda r: (-r['source_support_distance_ratio'], r['target_id'])
    selected = [r for center in sorted(grouped) for r in sorted(grouped[center], key=key)[2:4]]
    p.require(len(selected) == 110 and len({r['target_id'] for r in selected}) == 110
              and not ({r['target_id'] for r in selected} & excluded), 'Target cohort changed or overlaps earlier checks')
    selected = [{**row, 'runtime_target_id': _load_target_v3(
        None, row['target_performance'], 'ID', 'UNKNOWN')['target_id']} for row in selected]
    p.require(len({r['runtime_target_id'] for r in selected}) == 110, 'Runtime target IDs are not unique')
    return selected


def dependencies(root):
    paths = set(p.code_hashes(root)) | {CODE, REGISTRY, PREVIOUS_12, PREVIOUS_98, p.CONFIG}
    return {name: p.sha(root / name) for name in sorted(paths)}


def cohort(root, support, norm):
    rows = [json.loads(line) for line in (root / REGISTRY).read_text().splitlines()]
    excluded = {r['target_id'] for name in (PREVIOUS_12, PREVIOUS_98) for r in p.read(root / name)}
    p.require(len(excluded) == 110, 'Earlier check identity changed')
    return choose_targets(rows, support, norm, excluded)


def freeze(root, output):
    p.require(not output.exists(), 'Refusing to overwrite comparison output')
    cfg, _, _, support, norm, _ = p.source(root)
    targets = cohort(root, support, norm)
    output.mkdir(parents=True)
    p.write(output / 'targets.json', targets)
    p.write(output / 'protocol_lock.json', {
        'policy': POLICY, 'dependencies': dependencies(root), 'pipeline_protocol': cfg,
        'targets_sha256': p.sha(output / 'targets.json'),
    })
    print(json.dumps({'status': 'FROZEN', 'target_count': len(targets), 'output': str(output)}), flush=True)


def check_lock(root, output, *, current=True):
    lock = p.read(output / 'protocol_lock.json')
    p.require(lock['policy'] == POLICY and p.sha(output / 'targets.json') == lock['targets_sha256'], 'Protocol lock changed')
    if current:
        cfg, _, _, support, norm, _ = p.source(root)
        p.require(lock['dependencies'] == dependencies(root) and lock['pipeline_protocol'] == cfg,
                  'Comparison dependencies changed')
        p.require(p.read(output / 'targets.json') == cohort(root, support, norm), 'Frozen target selection changed')
    return lock


def rank(raw, cap=128):
    valid = sorted((r for r in raw if r['status'] == 'geometric_candidate'),
                   key=lambda r: (r['predicted_rho'], r['design_id'], r['raw_slot']))
    selected, seen = [], set()
    for row in valid:
        if row['design_id'] in seen:
            continue
        seen.add(row['design_id'])
        selected.append({**row, 'rank': len(selected)})
        if len(selected) == cap:
            break
    return selected


def verify_b(case, target, cfg):
    raw, selected, route = (p.read(case / name) for name in ('raw.json', 'selected.json', 'route.json'))
    p.require(route['target_id'] == target['target_id'] and len(raw) == (256 if route['status'] == 'routable' else 0),
              'B route or raw count changed')
    p.require(selected == rank(raw) and len(selected) <= POLICY['selected_candidate_cap'], 'B ranking changed')
    p.require(all(r['target_id'] == target['target_id'] and r['physics_bundle_hash'] == cfg['identity']['physics_bundle_hash']
                  for r in selected), 'B target or backend changed')
    return selected, route


def verify_ledger(case, selected, target, cfg):
    ledger = case / 'ledger'
    files = sorted(ledger.glob('*.json')) if ledger.exists() else []
    p.require(files == [ledger / f'{i:03d}.json' for i in range(len(files))], 'Ledger is not a sealed prefix')
    p.require(len(files) <= len(selected), 'Ledger exceeds B')
    results = []
    for i, path in enumerate(files):
        item = p.read(path)
        row, result = item['candidate'], item['evaluation']
        p.require(row == selected[i] and result['design_id'] == row['design_id']
                  and result['design'] == row['design'] and result['route_center_id'] == row['route_center_id'],
                  'Ledger candidate changed')
        rho = p._rho(result['performance'], target, cfg['relative_tolerances'])
        p.require(result['rho'] == rho and result['software_hit_5pct'] == (rho is not None and rho <= 1),
                  'Ledger score changed')
        p.require(result['formal_use_allowed'] is False and result['deployment_allowed'] is False,
                  'Ledger usage flags changed')
        results.append(result)
    return results


def arm(results, width):
    group = results[:width]
    best = p.best_result(group)
    return {'window': width, 'evaluations': len(group), 'diagnostic_valid_hits': sum(p.diagnostic_hit(r) for r in group),
            'hit': any(p.diagnostic_hit(r) for r in group), 'solver_converged': sum(r['solver_converged'] for r in group),
            'diagnostic_domain_valid': sum(r['diagnostic_domain_valid'] for r in group),
            'feasible': sum(r['feasible'] for r in group), 'best_design_id': best['design_id'] if best else None,
            'best_rho': best['rho'] if best else None, 'delivery_status': p.final_summary(best)['status']}


def verify_case(case, target, cfg):
    selected, route = verify_b(case, target, cfg)
    results = verify_ledger(case, selected, target, cfg)
    p.require(len(results) == len(selected), 'Case evaluation ledger incomplete')
    p.require(not (case / 'attempt.json').exists(), 'Unsealed physics attempt exists')
    f = p.read(case / 'final.json')
    p.require(f['arms'] == {'C8': arm(results, 8), 'C128': arm(results, 128)}, 'Case arms changed')
    for key, width in (('C8', 8), ('C128', 128)):
        best = p.best_result(results[:width])
        p.require(f['recompute'][key] == (best if best else None), 'F recompute changed')
        recompute_path = case / f'recompute_{key}.json'
        p.require((recompute_path.exists() and p.read(recompute_path) == best) if best else not recompute_path.exists(),
                  'F recompute artifact changed')
    p.require(f['target_id'] == target['target_id'] and f['candidate_count'] == len(selected)
              and f['route_status'] == route['status'], 'Case summary changed')
    return f


def run_case(root, case, row, cfg, local, support, norm, checkpoint):
    target = _load_target_v3(None, row['target_performance'], 'ID', 'UNKNOWN')
    p.require(target['target_id'] == row['runtime_target_id'], 'Runtime target ID changed')
    if (case / 'final.json').exists():
        return verify_case(case, target, cfg)
    case.mkdir(parents=True, exist_ok=True)
    if not (case / 'selected.json').exists():
        p.require(not any(case.iterdir()), 'Unsealed B exists; preserve and stop')
        old, raw, route = p.generate(root, target, cfg, local, support, norm, checkpoint)
        selected = rank(raw)
        p.require(old == selected[:len(old)] and len(old) == min(24, len(selected)), 'B prefix diverges from guided')
        p.write(case / 'raw.json', raw)
        p.write(case / 'route.json', route)
        p.write(case / 'selected.json', selected)
    selected, route = verify_b(case, target, cfg)
    results = verify_ledger(case, selected, target, cfg)
    attempt = case / 'attempt.json'
    p.require(not attempt.exists(), 'Unsealed physics attempt exists; preserve and stop')
    runtime = p.build_v3_development_physics_runtime(root)
    profile = p.ConstraintProfile.from_yaml(root / cfg['sources']['diagnostic_profile_path'])
    (case / 'ledger').mkdir(exist_ok=True)
    for i in range(len(results), len(selected)):
        p.write(attempt, {'phase': 'C', 'rank': i, 'design_id': selected[i]['design_id']})
        result = p.evaluate_design(selected[i], runtime, profile, target, cfg)
        p.write(case / 'ledger' / f'{i:03d}.json', {'candidate': selected[i], 'evaluation': result})
        attempt.unlink()
        results.append(result)
    arms = {'C8': arm(results, 8), 'C128': arm(results, 128)}
    recompute = {}
    for key, width in (('C8', 8), ('C128', 128)):
        best = p.best_result(results[:width])
        recompute_path = case / f'recompute_{key}.json'
        if best:
            if recompute_path.exists():
                repeated = p.read(recompute_path)
            else:
                p.write(attempt, {'phase': 'F', 'arm': key, 'design_id': best['design_id']})
                fresh = p.build_v3_development_physics_runtime(root)
                repeated = p.evaluate_design(best, fresh, profile, target, cfg)
                p.write(recompute_path, repeated)
                attempt.unlink()
            p.require(repeated == best, 'F independent recompute differs')
        else:
            p.require(not recompute_path.exists(), 'Unexpected F recompute')
        recompute[key] = repeated if best else None
    final = {'target_id': target['target_id'], 'source_center_id': row['source_center_id'],
             'candidate_count': len(selected), 'route_status': route['status'],
             'arms': arms, 'recompute': recompute,
             'first_valid_hit_rank': next((i + 1 for i, r in enumerate(results) if p.diagnostic_hit(r)), None)}
    p.write(case / 'final.json', final)
    return verify_case(case, target, cfg)


def summarize(reports):
    rescued = [r for r in reports if not r['arms']['C8']['hit'] and r['arms']['C128']['hit']]
    return {'status': 'COMPLETE', 'target_count': len(reports),
            'C8_hits': sum(r['arms']['C8']['hit'] for r in reports),
            'C128_hits': sum(r['arms']['C128']['hit'] for r in reports),
            'rescued_by_ranks_9_128': len(rescued),
            'rescue_rank_bands': {name: sum(lo <= r['first_valid_hit_rank'] <= hi for r in rescued)
                                  for name, lo, hi in [('9_24', 9, 24), ('25_64', 25, 64), ('65_128', 65, 128)]},
            'C_evaluations': sum(r['candidate_count'] for r in reports),
            'C8_evaluations': sum(r['arms']['C8']['evaluations'] for r in reports),
            'C128_evaluations': sum(r['arms']['C128']['evaluations'] for r in reports),
            'abstained': sum(r['route_status'] != 'routable' for r in reports),
            'short_of_128': sum(r['candidate_count'] < 128 for r in reports),
            'formal_use_allowed': False, 'deployment_allowed': False}


def run(root, output, *, verify_only=False, max_cases=None):
    check_lock(root, output)
    if not verify_only:
        p.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '6' and torch.cuda.is_available()
                  and torch.cuda.device_count() == 1, 'Comparison requires physical GPU 6')
        torch.use_deterministic_algorithms(True)
        print(json.dumps({'physical_gpu': 6, 'logical_device': 0, 'name': torch.cuda.get_device_name(0)}), flush=True)
    cfg, _, local, support, norm, checkpoint = p.source(root)
    reports = []
    targets = p.read(output / 'targets.json')
    p.require(max_cases is None or (not verify_only and 1 <= max_cases <= len(targets)), 'Invalid case limit')
    with p.run_lock(output):
        for index, row in enumerate(targets[:max_cases], 1):
            case = output / 'cases' / row['runtime_target_id']
            if verify_only:
                report = verify_case(case, _load_target_v3(None, row['target_performance'], 'ID', 'UNKNOWN'), cfg)
            else:
                report = run_case(root, case, row, cfg, local, support, norm, checkpoint)
            reports.append(report)
            print(json.dumps({'case': index, 'total': len(targets),
                              'target_id': report['target_id'], 'C8': report['arms']['C8']['hit'],
                              'C128': report['arms']['C128']['hit'], 'candidates': report['candidate_count']}), flush=True)
        if max_cases is not None and max_cases < len(targets):
            partial = summarize(reports)
            partial['status'] = 'PARTIAL'
            return {'status': 'PARTIAL', 'completed_cases': len(reports), 'total_cases': len(targets),
                    'partial_summary': partial}
        summary = summarize(reports)
        if verify_only:
            p.require(p.read(output / 'summary.json') == summary, 'Summary changed')
        else:
            p.write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'verify'))
    parser.add_argument('--output', type=Path, default=ROOT / OUTPUT)
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    root = ROOT
    if args.command == 'freeze':
        freeze(root, args.output)
    else:
        print(json.dumps(run(root, args.output, verify_only=args.command == 'verify', max_cases=args.max_cases), indent=2))
