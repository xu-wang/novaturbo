"""Expanded frozen development screening at the unchanged production C=8 window."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.ai import geometry22_pipeline_v3_guided as p
from src.ai.geometry19_pipeline_artifact import verify_completed_artifact

CODE = "src/ai/v3_refinement_expanded_check.py"
REGISTRY = "data/validation/v3_22d_extended_gate_l6f_gpu6_v1/target_registry.jsonl"
PREVIOUS_TARGETS = "data/validation/v3_refinement_difficult_check_v1/targets.json"
DEFAULT = "data/validation/v3_refinement_expanded_check_v1"
POLICY = {
    "version": "v3_refinement_expanded_check_v1",
    "scope": "development_natural_D_activation_not_independent_generalization",
    "target_count": 98,
    "selection": "tuning_only_within_source_fit_box_and_p95_ball;two_farthest_per_each_of_55_centers;exclude_previous12;distance_ratio_desc_then_target_id",
    "stopping": "complete_entire_frozen_queue_including_all_C_misses_no_success_based_stop",
    "pipeline_profile": p.PROFILE,
    "budgets": {"C": 8, "D": 8, "E": 8, "F": 1, "control": 8, "control_F": 1},
    "control": "same_B_and_C;evaluate_B_ranks_9_to_16_only_if_C_has_no_valid_hit;independent_F",
    "D_trigger": "unchanged_production_no_diagnostic_domain_valid_software_hit_in_C",
    "physical_gpu": 6,
    "no_retraining": True,
    "no_new_target_labels": True,
    "formal_use_allowed": False,
    "deployment_allowed": False,
}


def choose_targets(rows, support, norm, excluded):
    by_center = {s['center_id']: s for s in support}
    chosen = {}
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
        selected = {**row, 'source_support_distance_ratio': distance / radius}
        chosen.setdefault(row['source_center_id'], []).append(selected)
    p.require(len(chosen) == 55 and all(len(v) >= 2 for v in chosen.values()), 'Need two eligible targets per development center')
    key = lambda r: (-r['source_support_distance_ratio'], r['target_id'])
    representatives = [r for group in chosen.values() for r in sorted(group, key=key)[:2]]
    result = sorted((r for r in representatives if r['target_id'] not in excluded), key=key)
    p.require(len(excluded) == 12 and len(result) == 98 and len({r['target_id'] for r in result}) == 98,
              'Expanded queue must contain 98 distinct new development targets')
    return result


def dependencies(root):
    return {name: p.sha(root / name) for name in sorted(set(p.code_hashes(root)) | {CODE, REGISTRY, PREVIOUS_TARGETS, p.CONFIG})}


def freeze(root, output):
    p.require(not output.exists(), 'Refusing to overwrite frozen check')
    cfg, _, _, support, norm, _ = p.source(root)
    rows = [json.loads(s) for s in (root / REGISTRY).read_text().splitlines()]
    excluded = {r['target_id'] for r in p.read(root / PREVIOUS_TARGETS)}
    targets = choose_targets(rows, support, norm, excluded)
    output.mkdir(parents=True)
    p.write(output / 'targets.json', targets)
    p.write(output / 'protocol_lock.json', {'policy': POLICY, 'dependencies': dependencies(root),
        'pipeline_protocol': cfg, 'targets_sha256': p.sha(output / 'targets.json')})
    print(json.dumps({'status': 'FROZEN', 'target_count': len(targets), 'output': str(output)}), flush=True)


def check_lock(root, output, *, current=True):
    lock = p.read(output / 'protocol_lock.json')
    p.require(lock['policy'] == POLICY, 'D check policy changed')
    p.require(p.sha(output / 'targets.json') == lock['targets_sha256'], 'D check target queue changed')
    if current:
        p.require(lock['dependencies'] == dependencies(root), 'D check dependencies changed')
        cfg, _, _, support, norm, _ = p.source(root)
        p.require(cfg == lock['pipeline_protocol'], 'D check pipeline protocol changed')
        rows = [json.loads(s) for s in (root / REGISTRY).read_text().splitlines()]
        excluded = {r['target_id'] for r in p.read(root / PREVIOUS_TARGETS)}
        p.require(p.read(output / 'targets.json') == choose_targets(rows, support, norm, excluded), 'D check selection changed')
    return lock


def case_report(run, control):
    c = p.read(run / 'C/evaluations.json')
    d = p.read(run / 'D/evaluations.json')
    e = p.read(run / 'E/evaluations.json')
    f = p.read(run / 'F/summary.json')
    c_hit = any(p.diagnostic_hit(r) for r in c)
    d_hit = any(p.diagnostic_hit(r) for r in d)
    control_hit = any(p.diagnostic_hit(r) for r in control['evaluations'])
    return {'target_id': p.read(run / 'target.json')['target_id'],
            'route_status': p.read(run / 'B/route.json')['status'],
            'C_valid_hit': c_hit, 'D_triggered': bool(c) and not c_hit,
            'D_evaluations': len(d), 'D_valid_rescue': not c_hit and d_hit,
            'control_evaluations': len(control['evaluations']),
            'control_valid_rescue': not c_hit and control_hit,
            'D_only_rescue': not c_hit and d_hit and not control_hit,
            'E_valid_rescue': not c_hit and not d_hit and any(p.diagnostic_hit(r) for r in e),
            'D_best_rho': min((r['rho'] for r in d if r['diagnostic_domain_valid'] and r['rho'] is not None), default=None),
            'control_best_rho': min((r['rho'] for r in control['evaluations'] if r['diagnostic_domain_valid'] and r['rho'] is not None), default=None),
            'production_calls_including_F': len(c) + len(d) + len(e) + int(f['independent_recompute'] is not None),
            'additional_control_calls_including_F': len(control['evaluations']) + int(control['independent_recompute'] is not None),
            'production_F_domain_valid': bool(f['independent_recompute'] and f['independent_recompute']['diagnostic_domain_valid']),
            'production_feasible': bool(f['independent_recompute'] and f['independent_recompute']['feasible'])}


def control_run(root, run, cfg):
    c = p.read(run / 'C/evaluations.json')
    control = {'evaluations': [], 'independent_recompute': None}
    if not c or any(p.diagnostic_hit(r) for r in c):
        return control
    target = p.read(run / 'target.json')
    runtime = p.build_v3_development_physics_runtime(root)
    profile = p.ConstraintProfile.from_yaml(root / cfg['sources']['diagnostic_profile_path'])
    for row in p.read(run / 'B/candidates.json')[8:16]:
        control['evaluations'].append(p.evaluate_design(row, runtime, profile, target, cfg))
    best = p.best_result(c + control['evaluations'], cfg['f_recompute'])
    if best:
        repeated = p.evaluate_design(best, p.build_v3_development_physics_runtime(root), profile, target, cfg)
        p.require(repeated == best, 'D control F recompute differs')
        control['independent_recompute'] = repeated
    return control


def summarize(reports):
    return {'status': 'COMPLETED', 'scope': POLICY['scope'], 'target_count': len(reports),
            **{key: sum(r[key] for r in reports) for key in (
                'C_valid_hit', 'D_triggered', 'D_evaluations', 'D_valid_rescue',
                'control_evaluations', 'control_valid_rescue', 'D_only_rescue', 'E_valid_rescue',
                'production_calls_including_F', 'additional_control_calls_including_F', 'production_feasible')},
            'abstained_count': sum(r['route_status'] != 'routable' for r in reports),
            'cases': reports, 'physics_validation_status': 'not_completed',
            'formal_use_allowed': False, 'deployment_allowed': False}


def run(root, output):
    lock = check_lock(root, output)
    p.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '6' and torch.cuda.is_available()
              and torch.cuda.device_count() == 1, 'D check requires physical GPU 6')
    torch.use_deterministic_algorithms(True)
    print(json.dumps({'physical_gpu': 6, 'logical_gpu': 0, 'device': torch.cuda.get_device_name(0)}), flush=True)
    if (output / 'complete_manifest.json').exists():
        return verify(root, output)
    reports = []
    with p.run_lock(output):
        for index, row in enumerate(p.read(output / 'targets.json')):
            case = output / 'cases' / row['target_id']
            production = case / 'run'
            p.run_v3_routed_pipeline(root, target_values=row['target_performance'],
                                     output_root=production, resume=production.exists())
            verify_completed_artifact(production, root)
            control_path = case / 'control.json'
            if not control_path.exists():
                # A crash during control must preserve partial work and stop, not silently repeat it.
                marker = case / 'control_started.json'
                p.require(not marker.exists(), 'Unsealed control attempt exists; preserve and stop')
                p.write(marker, {'status': 'STARTED'})
                p.write(control_path, control_run(root, production, lock['pipeline_protocol']))
            report = case_report(production, p.read(control_path))
            reports.append(report)
            print(json.dumps({'case_index': index + 1, **report}), flush=True)
        summary = summarize(reports)
        p.write(output / 'summary.json', summary)
        files = {str(path.relative_to(output)): p.sha(path) for path in output.rglob('*')
                 if path.is_file() and path.name not in {'.runner.lock', 'complete_manifest.json'}}
        p.write(output / 'complete_manifest.json', {'files': files})
    verify(root, output)
    return summary


def verify(root, output):
    check_lock(root, output, current=False)
    files = p.read(output / 'complete_manifest.json')['files']
    actual = {str(path.relative_to(output)): p.sha(path) for path in output.rglob('*')
              if path.is_file() and path.name not in {'.runner.lock', 'complete_manifest.json'}}
    p.require(files == actual, 'D check artifacts changed')
    reports = []
    for row in p.read(output / 'targets.json'):
        case = output / 'cases' / row['target_id']
        verify_completed_artifact(case / 'run', root)
        reports.append(case_report(case / 'run', p.read(case / 'control.json')))
    summary = summarize(reports)
    p.require(summary == p.read(output / 'summary.json'), 'D check summary changed')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'verify'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve() if args.output else root / DEFAULT
    result = {'freeze': freeze, 'run': run, 'verify': verify}[args.action](root, output)
    if result:
        print(json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
