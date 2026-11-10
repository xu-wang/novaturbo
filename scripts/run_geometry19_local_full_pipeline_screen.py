#!/usr/bin/env python3
"""Frozen G4 rank-one-miss cohort; reuse the sealed B--F implementation."""
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

METRICS = 'data/validation/v2_2_performance_gated_local_cfm_v1/closure/g4_rank128_v1/Task_Metrics.jsonl'
CONFIG = 'config/experiments/geometry19_local_full_pipeline_difficult_v1.yaml'
OUTPUT = 'data/validation/v2_2_local_full_pipeline_v1/n3_rank1_miss_screen_v1'


def select_targets(rows):
    return sorted({row['target_index'] for row in rows
                   if row['training_seed'] == 42 and row['arm'] == 'gated_top1_local_cfm'
                   and not row['rank1_hit']})


def run(verify_only=False, resume=False):
    rows = [json.loads(line) for line in (ROOT / METRICS).read_text().splitlines()]
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    config.update(targets=select_targets(rows), output=OUTPUT,
                  version='geometry19_local_full_pipeline_rank1_miss_screen_v1')
    require(len(config['targets']) == 12, 'Frozen cohort changed')
    data = g3.preflight(ROOT)
    code = [CONFIG, METRICS, str(Path(__file__).relative_to(ROOT)),
            'tests/test_geometry19_local_full_pipeline_screen.py',
            'src/ai/geometry19_local_full_pipeline_entry.py',
            'src/ai/geometry19_local_full_pipeline.py']
    identity = dict(config=config, code={p: file_hash(ROOT / p) for p in code},
                    inputs=data['inputs'], runtime=data['context'].contract_mapping(),
                    selection='all G4 seed42 gated_top1_local_cfm rank1 misses; no new outcome selection')
    output = ROOT / OUTPUT
    if resume or verify_only:
        lock = json.loads((output / 'Protocol_Lock.json').read_text())
        require(all(lock[k] == v for k, v in identity.items()), 'Campaign identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        lock = dict(identity, created_at=datetime.now(timezone.utc).isoformat())
        g3.seal_json(output / 'Protocol_Lock.json', lock)
    lock_hash = g3.digest(lock)
    with (output / '.runner.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summaries = []
        models = None
        for index in config['targets']:
            directory = output / f'target{index:03d}'
            if directory.exists():
                require((directory / 'Target_Manifest.json').is_file(), 'Partial target: preserve and stop')
                summaries.append(verify_target(directory, lock_hash))
                continue
            require(not verify_only, 'Campaign incomplete')
            if models is None:
                g3.adapter._cuda_manifest()
                models, _ = g3.adapter.load_seed_models(ROOT, data['inference'], 42, data['decoders'], 'cuda:0')
                scorer_ids, scorer = g3._load_model(data['paths']['s2_oracle_local_ridge'], 'stable_38_oracle_root_local_ridge')
            summary = run_target(data, config, models, scorer_ids, scorer, index, directory, lock_hash)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        summary = dict(status='N3_RANK1_MISS_SCREEN_COMPLETED', targets=summaries,
                       first_hit_stage_counts=dict(Counter(str(r['first_hit_stage']) for r in summaries)),
                       costs=dict(sum((Counter(r['costs']) for r in summaries), Counter())),
                       all_delivery_pass=all(r['status'] == 'PASS' for r in summaries),
                       production_entry_integrated=False)
        if verify_only:
            require(json.loads((output / 'Summary.json').read_text()) == summary, 'Summary changed')
        else:
            g3.seal_json(output / 'Summary.json', summary)
        manifest = dict(lock_sha256=lock_hash, files={str(p.relative_to(output)): file_hash(p)
                        for p in output.rglob('*') if p.is_file()
                        and p.name not in {'.runner.lock', 'Complete_Manifest.json'}})
        if verify_only:
            require(json.loads((output / 'Complete_Manifest.json').read_text()) == manifest, 'Manifest changed')
        else:
            g3.seal_json(output / 'Complete_Manifest.json', manifest)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.verify_only, args.resume), indent=2))
