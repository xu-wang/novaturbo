#!/usr/bin/env python3
"""Frozen C-window and E early-stop comparison on the prior difficult cohort."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai import geometry19_performance_gated_g3 as g3
from src.ai.geometry19_local_full_pipeline_entry import run_target, verify_target
from src.ai.geometry19_performance_gated_protocol import file_hash, require
from scripts.run_geometry19_local_full_pipeline_screen import select_targets

PROTOCOL = 'config/experiments/geometry19_v2_2_c_window_comparison_v1.yaml'


def run(*, verify_only=False, resume=False):
    protocol = yaml.safe_load((ROOT / PROTOCOL).read_text())
    source_config = yaml.safe_load((ROOT / protocol['source_config']).read_text())
    rows = [json.loads(line) for line in (ROOT / protocol['source_metrics']).read_text().splitlines()]
    targets = select_targets(rows)
    require(targets == protocol['expected_targets'], 'Frozen cohort changed')
    data = g3.preflight(ROOT)
    code_paths = [PROTOCOL, protocol['source_config'], protocol['source_metrics'],
                  'scripts/run_geometry19_v22_c_window_comparison.py',
                  'scripts/run_geometry19_local_full_pipeline_screen.py',
                  'src/ai/geometry19_local_full_pipeline_entry.py',
                  'src/ai/geometry19_local_full_pipeline.py']
    identity = {
        'protocol': protocol,
        'code': {path: file_hash(ROOT / path) for path in code_paths},
        'inputs': data['inputs'], 'runtime': data['context'].contract_mapping(),
        'baseline': {key: file_hash(ROOT / protocol[key])
                     for key in ('sequential_baseline', 'c1_baseline')},
    }
    output = ROOT / protocol['output']
    if resume or verify_only:
        lock = json.loads((output / 'Protocol_Lock.json').read_text())
        require(all(lock[key] == value for key, value in identity.items()), 'Comparison identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        lock = {**identity, 'created_at': datetime.now(timezone.utc).isoformat()}
        g3.seal_json(output / 'Protocol_Lock.json', lock)
    lock_hash = g3.digest(lock)
    with (output / '.runner.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models = None
        arms = {}
        for window in protocol['c_windows']:
            config = dict(source_config)
            config['version'] = f"{protocol['version']}_c{window}"
            config['targets'] = targets
            config['budgets'] = {**source_config['budgets'], 'C': window}
            config['nsga2'] = {**source_config['nsga2'],
                               'stop_on_first_hit': protocol['e_stop_on_first_hit']}
            summaries = []
            for index in targets:
                directory = output / f'c{window}' / f'target{index:03d}'
                if directory.exists():
                    require((directory / 'Target_Manifest.json').is_file(),
                            'Partial target: preserve and stop')
                    summary = verify_target(directory, lock_hash)
                else:
                    require(not verify_only, 'Comparison incomplete')
                    if models is None:
                        g3.adapter._cuda_manifest()
                        models, _ = g3.adapter.load_seed_models(
                            ROOT, data['inference'], 42, data['decoders'], 'cuda:0')
                        scorer_ids, scorer = g3._load_model(
                            data['paths']['s2_oracle_local_ridge'],
                            'stable_38_oracle_root_local_ridge')
                    summary = run_target(data, config, models, scorer_ids, scorer,
                                         index, directory, lock_hash)
                    print(json.dumps({'window': window, **summary}), flush=True)
                summaries.append(summary)
            costs = sum((Counter(item['costs']) for item in summaries), Counter())
            arms[f'C{window}_E_early_stop'] = {
                'targets': summaries,
                'delivered': sum(item['status'] == 'PASS' for item in summaries),
                'first_hit_stage_counts': dict(Counter(str(item['first_hit_stage'])
                                                       for item in summaries)),
                'costs': dict(costs),
                'physical_calls_with_F': sum(value for name, value in costs.items()
                                             if name.endswith('_calls')),
            }
        baseline = json.loads((ROOT / protocol['sequential_baseline']).read_text())
        old = json.loads((ROOT / protocol['c1_baseline']).read_text())
        summary = {
            'status': 'V2_2_C_WINDOW_COMPARISON_COMPLETED',
            'formal_use_allowed': False,
            'targets': targets,
            'sequential_baseline_calls_with_F': baseline['aggregate']['sequential_delivery_calls'],
            'c1_baseline_calls_with_F': sum(
                value for name, value in old['costs'].items() if name.endswith('_calls')),
            'arms': arms,
        }
        if verify_only:
            require(json.loads((output / 'Summary.json').read_text()) == summary,
                    'Comparison summary changed')
        else:
            g3.seal_json(output / 'Summary.json', summary)
        manifest = {
            'lock_sha256': lock_hash,
            'files': {str(path.relative_to(output)): file_hash(path)
                      for path in output.rglob('*') if path.is_file()
                      and path.name not in {'.runner.lock', 'Complete_Manifest.json'}},
        }
        if verify_only:
            require(json.loads((output / 'Complete_Manifest.json').read_text()) == manifest,
                    'Comparison manifest changed')
        else:
            g3.seal_json(output / 'Complete_Manifest.json', manifest)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(verify_only=args.verify_only, resume=args.resume), indent=2))
