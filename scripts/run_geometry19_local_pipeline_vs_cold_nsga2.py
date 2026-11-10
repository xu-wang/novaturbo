#!/usr/bin/env python3
"""Compare V2.2 local B--F framework with pure cold-start NSGA-II."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai import geometry19_nsga2 as nsga
from src.ai import geometry19_performance_gated_g3 as g3
from src.ai.geometry19_inverse_contract import DiagnosticAcceptance, Geometry19InverseTarget
from src.ai.geometry19_local_full_pipeline import build_target_mapping
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, require, within
from src.ai.geometry19_physics import _json_safe, evaluate_geometry19_candidate
from src.physics import Environment

CONFIG = 'config/experiments/geometry19_local_pipeline_vs_cold_nsga2_v1.yaml'


def project_counts(summary):
    discovery = sum(value for key, value in summary['costs'].items()
                    if key.endswith('_calls') and key != 'F_calls')
    return discovery, discovery + summary['costs'].get('F_calls', 0)


def aggregate(rows):
    project_hits = sum(row['project_hit'] for row in rows)
    cold_hits = sum(row['cold_nsga2_hit'] for row in rows)
    comparable = [row for row in rows if row['project_hit'] and row['cold_nsga2_hit']]
    project_calls = sum(row['project_discovery_calls'] for row in comparable)
    cold_calls = sum(row['cold_nsga2_discovery_calls'] for row in comparable)
    wins = Counter(
        'project' if row['project_discovery_calls'] < row['cold_nsga2_discovery_calls']
        else 'cold_nsga2' if row['project_discovery_calls'] > row['cold_nsga2_discovery_calls']
        else 'tie' for row in comparable
    )
    return {
        'tasks': len(rows), 'project_hits': project_hits, 'cold_nsga2_hits': cold_hits,
        'both_hit_tasks': len(comparable),
        'project_only_hits': sum(row['project_hit'] and not row['cold_nsga2_hit'] for row in rows),
        'cold_nsga2_only_hits': sum(row['cold_nsga2_hit'] and not row['project_hit'] for row in rows),
        'neither_hits': sum(not row['project_hit'] and not row['cold_nsga2_hit'] for row in rows),
        'both_hit_project_discovery_calls': project_calls,
        'both_hit_cold_nsga2_discovery_calls': cold_calls,
        'both_hit_calls_saved_by_project': cold_calls - project_calls,
        'both_hit_saved_fraction': None if cold_calls == 0 else (cold_calls - project_calls) / cold_calls,
        'paired_wins_on_both_hits': dict(wins),
        'cold_nsga2_capped_failed_calls': sum(row['cold_nsga2_discovery_calls'] for row in rows if not row['cold_nsga2_hit']),
    }


def run_cold_target(data, config, index, directory):
    public = data['runtime_targets'][index]
    target = Geometry19InverseTarget.from_mapping(build_target_mapping(**public['performance']))
    tolerances = data['parent']['evaluation']['relative_tolerances']
    acceptance = DiagnosticAcceptance(tolerances, dict(zip(g3.adapter.OUTPUT_NAMES, [1e-12] * 3)), False)
    fields = list(data['context'].physics_runtime.design_schema['design_variables'])
    lower = np.asarray([field['bounds'][0] for field in fields], dtype=float)
    upper = np.asarray([field['bounds'][1] for field in fields], dtype=float)
    integer_indices = [i for i, field in enumerate(fields) if field['dtype'] == 'int64']
    protocol = config['cold_nsga2']
    population_size = protocol['population']
    seed = protocol['seed']
    genes = nsga._lhs_fill(population_size, lower, upper, integer_indices, np.random.default_rng(seed))
    evolution_rng = np.random.default_rng(seed)
    cache = {}
    ledger = []
    physical_calls = 0
    hit = None

    def evaluate(vector, generation, population_index, source):
        nonlocal physical_calls, hit
        mapping = nsga._design_mapping(vector, fields)
        design_hash = nsga._design_hash(mapping)
        cache_hit = design_hash in cache
        if not cache_hit:
            event = {'event': 'call_started', 'call_index': physical_calls, 'generation': generation,
                     'population_index': population_index, 'design_hash': design_hash}
            with (directory / 'Calls.jsonl').open('ab') as stream:
                stream.write(canonical(event) + b'\n'); stream.flush()
                import os
                os.fsync(stream.fileno())
            result = evaluate_geometry19_candidate(
                physical_calls, f'cold_nsga2_generation_{generation}', mapping, design_hash,
                target, acceptance, data['context'].physics_runtime.model_config,
                data['context'].physics_runtime.constraint_profile, Environment(),
                physics_runtime=data['context'].physics_runtime,
            )
            result['inside_target_domain'] = bool(
                result['performance'] and data['context'].target_domain.assess(result['performance']).inside
            )
            result['physics_acceptable'] = result['physics_acceptable'] and result['inside_target_domain']
            result['development_diagnostic_hit'] = result['development_diagnostic_hit'] and result['physics_acceptable']
            if not result['physics_acceptable']:
                result['hit_alignment'] = None
            cache[design_hash] = _json_safe(result)
            physical_calls += 1
        result = dict(cache[design_hash])
        ledger.append({'generation': generation, 'population_index': population_index,
                       'source': source, 'cache_hit': cache_hit, 'genes': mapping, 'evaluation': result})
        if result['development_diagnostic_hit']:
            hit = result
        return {'genes': np.asarray(vector, dtype=float), 'evaluation': result}

    population = []
    for position, vector in enumerate(genes):
        population.append(evaluate(vector, 0, position, 'cold_start_lhs'))
        if hit is not None:
            break
    for generation in range(1, protocol['generations_after_initial'] + 1):
        if hit is not None or physical_calls >= protocol['maximum_physics_calls']:
            break
        require(len(population) == population_size, 'Evolution requires a complete initial population')
        ranks, crowding = nsga._rank_and_crowding(population)
        children = []
        while len(children) < population_size:
            left = population[nsga._tournament(ranks, crowding, evolution_rng)]['genes']
            right = population[nsga._tournament(ranks, crowding, evolution_rng)]['genes']
            for child in nsga._sbx(left, right, lower, upper, evolution_rng, rate=protocol['crossover_rate']):
                children.append(nsga._mutate(child, lower, upper, integer_indices, evolution_rng,
                                             rate=protocol['mutation_rate'], scale=protocol['mutation_scale']))
                if len(children) == population_size:
                    break
        offspring = []
        for position, vector in enumerate(children):
            offspring.append(evaluate(vector, generation, position, 'nsga2_offspring'))
            if hit is not None or physical_calls >= protocol['maximum_physics_calls']:
                break
        if hit is not None:
            break
        require(len(offspring) == population_size, 'Budget must cover a complete nonterminal generation')
        population = nsga._environmental_selection(population + offspring, population_size)
    g3.adapter._write_once(directory / 'Ledger.jsonl', b''.join(canonical(row) + b'\n' for row in ledger))
    summary = {
        'target_index': index, 'hit': hit is not None, 'first_hit_physics_call': physical_calls if hit else None,
        'physical_calls': physical_calls, 'logical_evaluations': len(ledger),
        'stopped_generation': ledger[-1]['generation'],
        'hit_design_hash': None if hit is None else hit['design_hash'],
    }
    g3.seal_json(directory / 'Summary.json', summary)
    g3.seal_json(directory / 'Manifest.json', {name: file_hash(directory / name)
                 for name in ('Calls.jsonl', 'Ledger.jsonl', 'Summary.json')})
    return summary


def verify_target(directory):
    manifest = json.loads((directory / 'Manifest.json').read_text())
    for name, digest in manifest.items():
        require(file_hash(directory / name) == digest, 'Cold NSGA-II target artifact changed')
    return json.loads((directory / 'Summary.json').read_text())


def run(*, resume=False, verify_only=False):
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    data = g3.preflight(ROOT)
    project_source = within(ROOT, config['project_source'])
    cost_source = within(ROOT, config['cost_source'])
    output = within(ROOT, config['output'])
    source_files = ['Summary.json', 'Protocol_Lock.json', 'Complete_Manifest.json']
    identity = {
        'config': config, 'config_sha256': file_hash(ROOT / CONFIG),
        'runner_sha256': file_hash(Path(__file__)),
        'project_source': {name: file_hash(project_source / name) for name in source_files},
        'cost_source_summary_sha256': file_hash(cost_source / 'Summary.json'),
        'nsga2_code_sha256': file_hash(ROOT / 'src/ai/geometry19_nsga2.py'),
        'runtime': data['context'].contract_mapping(),
    }
    if resume or verify_only:
        lock = json.loads((output / 'Protocol_Lock.json').read_text())
        require(all(lock[key] == value for key, value in identity.items()), 'Campaign identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        lock = {**identity, 'created_at': datetime.now(timezone.utc).isoformat()}
        g3.seal_json(output / 'Protocol_Lock.json', lock)
    project = {row['target_index']: row for row in json.loads((project_source / 'Summary.json').read_text())['targets']}
    with (output / '.runner.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for index in config['targets']:
            directory = output / f'target{index:03d}'
            if directory.exists():
                require((directory / 'Manifest.json').is_file(), 'Partial target exists; preserve and stop')
                cold = verify_target(directory)
            else:
                require(not verify_only, 'Campaign incomplete')
                directory.mkdir(parents=True)
                cold = run_cold_target(data, config, index, directory)
            project_discovery, project_delivery = project_counts(project[index])
            row = {
                'target_index': index, 'project_hit': project[index]['status'] == 'PASS',
                'project_discovery_calls': project_discovery, 'project_delivery_calls': project_delivery,
                'cold_nsga2_hit': cold['hit'], 'cold_nsga2_discovery_calls': cold['physical_calls'],
                'cold_nsga2_delivery_calls': cold['physical_calls'] + (1 if cold['hit'] else 0),
                'cold_nsga2_stopped_generation': cold['stopped_generation'],
            }
            rows.append(row); print(json.dumps(row), flush=True)
        summary = {'status': 'N5_PROJECT_VS_COLD_NSGA2_COMPLETED', 'targets': rows, 'aggregate': aggregate(rows)}
        if (output / 'Summary.json').exists():
            require(json.loads((output / 'Summary.json').read_text()) == summary, 'Summary changed')
        else:
            require(not verify_only, 'Summary missing'); g3.seal_json(output / 'Summary.json', summary)
        complete = {'files': {str(path.relative_to(output)): file_hash(path) for path in output.rglob('*')
                    if path.is_file() and path.name not in {'.runner.lock', 'Complete_Manifest.json'}}}
        if (output / 'Complete_Manifest.json').exists():
            require(json.loads((output / 'Complete_Manifest.json').read_text()) == complete, 'Manifest changed')
        else:
            require(not verify_only, 'Manifest missing'); g3.seal_json(output / 'Complete_Manifest.json', complete)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--resume', action='store_true'); parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(); print(json.dumps(run(resume=args.resume, verify_only=args.verify_only), indent=2))
