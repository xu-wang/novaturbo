#!/usr/bin/env python3
"""Run global and route-informed NSGA-II with 10x per-task project budgets."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import json
import math
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

CONFIG = 'config/experiments/geometry19_framework_vs_nsga2_10x_local_v1.yaml'


def project_discovery(summary):
    return sum(value for key, value in summary['costs'].items()
               if key.endswith('_calls') and key != 'F_calls')


def project_ball(matrix, center, radius):
    safe = np.clip(np.asarray(matrix, dtype=float), 0.0, 1.0)
    delta = safe - center
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    factor = np.minimum(1.0, radius / np.maximum(norm, 1e-20))
    return center + delta * factor


def local_initial(count, center, radius, rng):
    directions = rng.normal(size=(count, len(center)))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-20)
    radii = radius * rng.random(count) ** (1.0 / len(center))
    return project_ball(center + directions * radii[:, None], center, radius)


def aggregate(rows, arm):
    selected = [row for row in rows if row['arm'] == arm]
    hits = [row for row in selected if row['hit']]
    project = sum(row['project_discovery_calls'] for row in selected)
    calls = sum(row['physical_calls'] for row in selected)
    both = [row for row in hits if row['project_hit']]
    return {
        'tasks': len(selected), 'hits': len(hits), 'total_physics_calls': calls,
        'project_total_discovery_calls': project,
        'both_hit_tasks': len(both),
        'both_hit_nsga2_calls': sum(row['physical_calls'] for row in both),
        'both_hit_project_calls': sum(row['project_discovery_calls'] for row in both),
        'budget_exhausted_without_hit': sum(not row['hit'] for row in selected),
        'paired_wins_on_both_hits': dict(Counter(
            'project' if row['project_discovery_calls'] < row['physical_calls']
            else arm if row['project_discovery_calls'] > row['physical_calls'] else 'tie'
            for row in both)),
    }


def render_report(summary):
    lines = [
        '# 项目框架与10倍预算NSGA-II调用比较', '',
        f"日期：{datetime.now(timezone.utc).date().isoformat()}。首次命中调用不含F交付复算；每个NSGA-II任务的物理调用上限为该任务项目框架首次命中调用数的10倍。", '',
        '| target | 框架调用 | 10倍上限 | 全域NSGA-II | 同局部域NSGA-II |',
        '|---|---:|---:|---:|---:|',
    ]
    by_target = {}
    for row in summary['rows']:
        by_target.setdefault(row['target_index'], {})[row['arm']] = row
    for index in sorted(by_target):
        global_row = by_target[index]['global_cold_nsga2']
        local_row = by_target[index]['route_informed_local_nsga2']
        def value(row):
            return str(row['physical_calls']) if row['hit'] else f">{row['physical_calls']}（未命中）"
        lines.append(f"| {index} | {global_row['project_discovery_calls']} | {global_row['budget']} | {value(global_row)} | {value(local_row)} |")
    lines.extend(['', '## 汇总', ''])
    for arm, label in [('global_cold_nsga2', '纯全域cold-start NSGA-II'),
                       ('route_informed_local_nsga2', '相同gate局部域NSGA-II')]:
        row = summary['aggregate'][arm]
        lines.append(f"- {label}：{row['hits']}/{row['tasks']}命中，实际物理调用{row['total_physics_calls']}次；双方命中的{row['both_hit_tasks']}个任务中，NSGA-II调用{row['both_hit_nsga2_calls']}次，框架调用{row['both_hit_project_calls']}次。")
    lines.extend(['',
        '全域基线不使用生成器、代理、gate、D-Flow或梯度增强。局部域基线只借用框架已冻结的gate专家及其0.20支持域，并固定该专家整数变量；它不使用生成网络、代理排序、D-Flow或梯度增强，因此用于区分搜索域先验与完整框架的贡献。', '',
        '任务来自旧G4首位未命中开发集合，结果不代表未知目标总体成功率。未命中行是右删失结果，不能把预算耗尽次数表述为达到命中的成本。', '',
        f"证据：`{summary['output']}/Summary.json`、`Protocol_Lock.json`、`Complete_Manifest.json`及每个目标/方法的完整账本。", ''
    ])
    return '\n'.join(lines)


def run_arm(data, config, project, index, arm, directory):
    public = data['runtime_targets'][index]
    target = Geometry19InverseTarget.from_mapping(build_target_mapping(**public['performance']))
    tolerances = data['parent']['evaluation']['relative_tolerances']
    acceptance = DiagnosticAcceptance(tolerances, dict(zip(g3.adapter.OUTPUT_NAMES, [1e-12] * 3)), False)
    fields = list(data['context'].physics_runtime.design_schema['design_variables'])
    project_calls = project_discovery(project)
    budget = config['budget_multiplier'] * project_calls
    population_size = min(config['nsga2']['maximum_population'], budget)
    seed = config['nsga2']['seed']
    init_rng = np.random.default_rng(seed)
    evolution_rng = np.random.default_rng(seed)
    if arm == 'global_cold_nsga2':
        lower = np.asarray([field['bounds'][0] for field in fields], dtype=float)
        upper = np.asarray([field['bounds'][1] for field in fields], dtype=float)
        integer_indices = [i for i, field in enumerate(fields) if field['dtype'] == 'int64']
        genes = nsga._lhs_fill(population_size, lower, upper, integer_indices, init_rng)
        decode = lambda vector: nsga._design_mapping(vector, fields)
        constrain = lambda matrix: np.asarray(matrix)
    else:
        b = json.loads((within(ROOT, config['route_source']) / f'target{index:03d}' / 'runtime' / 'B.json').read_text())
        route = int(b['selected_expert'])
        expert = data['decoders']['experts'][route]
        center = np.asarray(expert['normalized_medoid_continuous'], dtype=float)
        radius = float(expert['radius']) * 0.20
        lower = np.zeros(len(center)); upper = np.ones(len(center)); integer_indices = []
        genes = local_initial(population_size, center, radius, init_rng)
        fixed = dict(zip(data['decoders']['integer_axes'], expert['integer_values']))
        axes = data['decoders']['continuous_axes']; lo = np.asarray(data['decoders']['continuous_lower']); span = np.asarray(data['decoders']['continuous_span'])
        decode = lambda vector: {**fixed, **{name: float(value) for name, value in zip(axes, lo + span * vector)}}
        constrain = lambda matrix: project_ball(matrix, center, radius)
    population = []; ledger = []; physical_calls = 0; cache = {}; hit = None

    def evaluate(vector, generation, position, source):
        nonlocal physical_calls, hit
        mapping = decode(vector); design_hash = nsga._design_hash(mapping); cache_hit = design_hash in cache
        if not cache_hit:
            event = {'event': 'call_started', 'call_index': physical_calls, 'generation': generation,
                     'population_index': position, 'design_hash': design_hash}
            with (directory / 'Calls.jsonl').open('ab') as stream:
                stream.write(canonical(event) + b'\n'); stream.flush()
                import os
                os.fsync(stream.fileno())
            result = evaluate_geometry19_candidate(
                physical_calls, arm, mapping, design_hash, target, acceptance,
                data['context'].physics_runtime.model_config, data['context'].physics_runtime.constraint_profile,
                Environment(), physics_runtime=data['context'].physics_runtime)
            result['inside_target_domain'] = bool(result['performance'] and data['context'].target_domain.assess(result['performance']).inside)
            result['physics_acceptable'] = result['physics_acceptable'] and result['inside_target_domain']
            result['development_diagnostic_hit'] = result['development_diagnostic_hit'] and result['physics_acceptable']
            if not result['physics_acceptable']: result['hit_alignment'] = None
            cache[design_hash] = _json_safe(result); physical_calls += 1
        result = dict(cache[design_hash])
        ledger.append({'generation': generation, 'population_index': position, 'source': source,
                       'cache_hit': cache_hit, 'genes': mapping, 'evaluation': result})
        if result['development_diagnostic_hit']: hit = result
        return {'genes': np.asarray(vector), 'evaluation': result}

    for position, vector in enumerate(genes):
        population.append(evaluate(vector, 0, position, 'initial'))
        if hit is not None or physical_calls >= budget: break
    generation = 0
    while hit is None and physical_calls < budget and len(population) == population_size:
        generation += 1; ranks, crowding = nsga._rank_and_crowding(population); children = []
        while len(children) < population_size:
            left = population[nsga._tournament(ranks, crowding, evolution_rng)]['genes']
            right = population[nsga._tournament(ranks, crowding, evolution_rng)]['genes']
            for child in nsga._sbx(left, right, lower, upper, evolution_rng, rate=config['nsga2']['crossover_rate']):
                child = nsga._mutate(child, lower, upper, integer_indices, evolution_rng,
                                     rate=config['nsga2']['mutation_rate'], scale=config['nsga2']['mutation_scale'])
                children.append(constrain(child.reshape(1, -1))[0])
                if len(children) == population_size: break
        offspring = []
        for position, vector in enumerate(children):
            offspring.append(evaluate(vector, generation, position, 'offspring'))
            if hit is not None or physical_calls >= budget: break
        if hit is not None or physical_calls >= budget: break
        population = nsga._environmental_selection(population + offspring, population_size)
    g3.adapter._write_once(directory / 'Ledger.jsonl', b''.join(canonical(row) + b'\n' for row in ledger))
    summary = {'target_index': index, 'arm': arm, 'project_hit': project['status'] == 'PASS',
               'project_discovery_calls': project_calls, 'budget': budget, 'hit': hit is not None,
               'physical_calls': physical_calls, 'logical_evaluations': len(ledger),
               'stopped_generation': generation, 'hit_design_hash': None if hit is None else hit['design_hash']}
    g3.seal_json(directory / 'Summary.json', summary)
    g3.seal_json(directory / 'Manifest.json', {name: file_hash(directory / name)
                 for name in ('Calls.jsonl', 'Ledger.jsonl', 'Summary.json')})
    return summary


def verify(directory):
    manifest = json.loads((directory / 'Manifest.json').read_text())
    for name, digest in manifest.items(): require(file_hash(directory / name) == digest, 'Arm artifact changed')
    return json.loads((directory / 'Summary.json').read_text())


def run(*, resume=False, verify_only=False):
    config = yaml.safe_load((ROOT / CONFIG).read_text()); data = g3.preflight(ROOT)
    project_root = within(ROOT, config['project_source']); output = within(ROOT, config['output'])
    identity = {'config': config, 'config_sha256': file_hash(ROOT / CONFIG), 'runner_sha256': file_hash(Path(__file__)),
                'project_summary_sha256': file_hash(project_root / 'Summary.json'),
                'route_complete_manifest_sha256': file_hash(within(ROOT, config['route_source']) / 'Complete_Manifest.json'),
                'nsga2_code_sha256': file_hash(ROOT / 'src/ai/geometry19_nsga2.py'), 'runtime': data['context'].contract_mapping()}
    if resume or verify_only:
        lock = json.loads((output / 'Protocol_Lock.json').read_text()); require(all(lock[k] == v for k, v in identity.items()), 'Identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False); lock = {**identity, 'created_at': datetime.now(timezone.utc).isoformat()}; g3.seal_json(output / 'Protocol_Lock.json', lock)
    project = {row['target_index']: row for row in json.loads((project_root / 'Summary.json').read_text())['targets']}
    with (output / '.runner.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB); rows = []
        for index in config['targets']:
            for arm in config['arms']:
                directory = output / f'target{index:03d}' / arm
                if directory.exists():
                    require((directory / 'Manifest.json').is_file(), 'Partial arm exists; preserve and stop'); row = verify(directory)
                else:
                    require(not verify_only, 'Campaign incomplete'); directory.mkdir(parents=True); row = run_arm(data, config, project[index], index, arm, directory)
                rows.append(row); print(json.dumps(row), flush=True)
        summary = {'status': 'N6_FRAMEWORK_VS_NSGA2_10X_LOCAL_COMPLETED', 'rows': rows,
                   'aggregate': {arm: aggregate(rows, arm) for arm in config['arms']}, 'output': config['output']}
        if (output / 'Summary.json').exists(): require(json.loads((output / 'Summary.json').read_text()) == summary, 'Summary changed')
        else: require(not verify_only, 'Summary missing'); g3.seal_json(output / 'Summary.json', summary)
        report = within(ROOT, config['report']); report.parent.mkdir(parents=True, exist_ok=True)
        if not verify_only: g3.adapter._write_once(report, render_report(summary).encode())
        require(report.is_file(), 'Report missing')
        complete = {'files': {str(path.relative_to(output)): file_hash(path) for path in output.rglob('*') if path.is_file() and path.name not in {'.runner.lock', 'Complete_Manifest.json'}},
                    'report': {config['report']: file_hash(report)}}
        if (output / 'Complete_Manifest.json').exists(): require(json.loads((output / 'Complete_Manifest.json').read_text()) == complete, 'Manifest changed')
        else: require(not verify_only, 'Manifest missing'); g3.seal_json(output / 'Complete_Manifest.json', complete)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--resume', action='store_true'); parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(); print(json.dumps(run(resume=args.resume, verify_only=args.verify_only), indent=2))
