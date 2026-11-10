#!/usr/bin/env python3
"""Paired cost comparison on the exact B candidates from the N3 screen."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai import geometry19_performance_gated_g3 as g3
from src.ai.geometry19_inverse_contract import DiagnosticAcceptance, Geometry19InverseTarget
from src.ai.geometry19_local_full_pipeline import build_target_mapping
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, require, within
from src.ai.geometry19_physics import _json_safe, evaluate_geometry19_candidate
from src.physics import Environment

CONFIG = 'config/experiments/geometry19_local_pipeline_cost_comparison_v1.yaml'


def summarize_target(index, evaluations, enhanced):
    hits = [rank for rank, row in enumerate(evaluations, 1) if row['development_diagnostic_hit']]
    first = hits[0] if hits else None
    sequential_discovery = first if first is not None else len(evaluations)
    sequential_delivery = sequential_discovery + (1 if first is not None else 0)
    enhanced_discovery = sum(value for key, value in enhanced['costs'].items() if key.endswith('_calls') and key != 'F_calls')
    enhanced_delivery = enhanced_discovery + enhanced['costs'].get('F_calls', 0)
    return {
        'target_index': index,
        'ranked_candidates_evaluated': len(evaluations),
        'missing_slots': 64 - len(evaluations),
        'first_hit_rank': first,
        'hit_count': len(hits),
        'hit_fraction': len(hits) / len(evaluations),
        'sequential_discovery_calls': sequential_discovery,
        'sequential_delivery_calls': sequential_delivery,
        'enhanced_first_hit_stage': enhanced['first_hit_stage'],
        'enhanced_discovery_calls': enhanced_discovery,
        'enhanced_delivery_calls': enhanced_delivery,
        'delivery_call_delta_enhanced_minus_sequential': enhanced_delivery - sequential_delivery,
    }


def aggregate(rows):
    count = len(rows)
    seq = sum(row['sequential_delivery_calls'] for row in rows)
    enhanced = sum(row['enhanced_delivery_calls'] for row in rows)
    wins = Counter('enhanced' if row['enhanced_delivery_calls'] < row['sequential_delivery_calls']
                   else 'sequential' if row['enhanced_delivery_calls'] > row['sequential_delivery_calls']
                   else 'tie' for row in rows)
    return {
        'tasks': count,
        'tasks_with_sequential_hit': sum(row['first_hit_rank'] is not None for row in rows),
        'sequential_delivery_calls': seq,
        'enhanced_delivery_calls': enhanced,
        'call_reduction': seq - enhanced,
        'relative_call_reduction': (seq - enhanced) / seq,
        'mean_sequential_delivery_calls': seq / count,
        'mean_enhanced_delivery_calls': enhanced / count,
        'paired_wins': dict(wins),
        'all_available_candidates_evaluated': all(
            row['ranked_candidates_evaluated'] + row['missing_slots'] == 64 for row in rows
        ),
    }


def evaluate_target(data, source, output, index):
    b = json.loads((source / f'target{index:03d}' / 'runtime' / 'B.json').read_text())
    by_hash = {row['design_hash']: row for row in b['candidates'] if row['valid']}
    require(len(b['ranking']) == len(by_hash) and 0 < len(by_hash) <= 64,
            'Ranked B candidates must be valid, unique, nonempty, and within raw slots')
    ranked = [by_hash[value] for value in b['ranking']]
    public = data['runtime_targets'][index]
    target = Geometry19InverseTarget.from_mapping(build_target_mapping(**public['performance']))
    tolerances = data['parent']['evaluation']['relative_tolerances']
    acceptance = DiagnosticAcceptance(tolerances, dict(zip(g3.adapter.OUTPUT_NAMES, [1e-12] * 3)), False)
    target_dir = output / f'target{index:03d}'
    require(not target_dir.exists(), 'Partial target exists; preserve and stop')
    target_dir.mkdir(parents=True)
    results = []
    calls = target_dir / 'Calls.jsonl'
    for rank, row in enumerate(ranked, 1):
        event = {'event': 'call_started', 'rank': rank, 'design_hash': row['design_hash']}
        with calls.open('ab') as stream:
            stream.write(canonical(event) + b'\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        result = evaluate_geometry19_candidate(
            rank - 1, 'sequential_B', row['geometry19'], row['design_hash'], target, acceptance,
            data['context'].physics_runtime.model_config,
            data['context'].physics_runtime.constraint_profile,
            Environment(), physics_runtime=data['context'].physics_runtime,
        )
        result['inside_target_domain'] = bool(result['performance'] and data['context'].target_domain.assess(result['performance']).inside)
        result['physics_acceptable'] = result['physics_acceptable'] and result['inside_target_domain']
        result['development_diagnostic_hit'] = result['development_diagnostic_hit'] and result['physics_acceptable']
        if not result['physics_acceptable']:
            result['hit_alignment'] = None
        results.append(_json_safe(result))
    payload = b''.join(canonical(row) + b'\n' for row in results)
    g3.adapter._write_once(target_dir / 'Evaluations.jsonl', payload)
    enhanced = json.loads((source / f'target{index:03d}' / 'Pipeline_Summary.json').read_text())
    summary = summarize_target(index, results, enhanced)
    g3.seal_json(target_dir / 'Summary.json', summary)
    manifest = {name: file_hash(target_dir / name) for name in ('Calls.jsonl', 'Evaluations.jsonl', 'Summary.json')}
    g3.seal_json(target_dir / 'Manifest.json', manifest)
    return summary


def verify_target(directory):
    manifest = json.loads((directory / 'Manifest.json').read_text())
    for name, digest in manifest.items():
        require(file_hash(directory / name) == digest, 'Target artifact changed')
    return json.loads((directory / 'Summary.json').read_text())


def run(*, resume=False, verify_only=False):
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    source = within(ROOT, config['source_campaign'])
    output = within(ROOT, config['output'])
    data = g3.preflight(ROOT)
    source_files = ['Protocol_Lock.json', 'Summary.json', 'Complete_Manifest.json']
    for index in config['targets']:
        source_files.extend([f'target{index:03d}/runtime/B.json', f'target{index:03d}/Pipeline_Summary.json'])
    identity = {
        'config': config,
        'config_sha256': file_hash(ROOT / CONFIG),
        'runner_sha256': file_hash(Path(__file__)),
        'source_files': {name: file_hash(source / name) for name in source_files},
        'runtime': data['context'].contract_mapping(),
        'comparison_note': 'All 64 B candidates are evaluated for coverage; operational cost stops at first hit.',
    }
    if resume or verify_only:
        lock = json.loads((output / 'Protocol_Lock.json').read_text())
        require(all(lock[key] == value for key, value in identity.items()), 'Comparison identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        lock = {**identity, 'created_at': datetime.now(timezone.utc).isoformat()}
        g3.seal_json(output / 'Protocol_Lock.json', lock)
    with (output / '.runner.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for index in config['targets']:
            directory = output / f'target{index:03d}'
            if directory.exists():
                require((directory / 'Manifest.json').is_file(), 'Partial target exists; preserve and stop')
                rows.append(verify_target(directory))
            else:
                require(not verify_only, 'Comparison is incomplete')
                row = evaluate_target(data, source, output, index)
                rows.append(row)
                print(json.dumps(row), flush=True)
        summary = {'status': 'N4_COST_COMPARISON_COMPLETED', 'targets': rows, 'aggregate': aggregate(rows)}
        if (output / 'Summary.json').exists():
            require(json.loads((output / 'Summary.json').read_text()) == summary, 'Summary changed')
        else:
            require(not verify_only, 'Summary missing')
            g3.seal_json(output / 'Summary.json', summary)
        files = {str(path.relative_to(output)): file_hash(path) for path in output.rglob('*')
                 if path.is_file() and path.name not in {'.runner.lock', 'Complete_Manifest.json'}}
        complete = {'files': files, 'protocol_lock_sha256': file_hash(output / 'Protocol_Lock.json')}
        if (output / 'Complete_Manifest.json').exists():
            require(json.loads((output / 'Complete_Manifest.json').read_text()) == complete, 'Complete manifest changed')
        else:
            require(not verify_only, 'Complete manifest missing')
            g3.seal_json(output / 'Complete_Manifest.json', complete)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(resume=args.resume, verify_only=args.verify_only), indent=2))
