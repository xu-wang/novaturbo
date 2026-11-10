#!/usr/bin/env python3
"""Run global and route-informed NSGA-II with 5,000 calls per task and arm."""
from copy import deepcopy
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
from src.ai.geometry19_performance_gated_protocol import file_hash, require, within
from scripts import run_geometry19_framework_vs_nsga2_10x_local as base

CONFIG = 'config/experiments/geometry19_framework_vs_nsga2_fixed5000_v1.yaml'


def task_config(config, project):
    """Adapt the frozen runner without changing its bound 10x experiment code."""
    result = deepcopy(config)
    calls = base.project_discovery(project)
    require(calls > 0, 'Project discovery calls must be positive')
    result['budget_multiplier'] = config['fixed_physics_call_budget_per_target_per_arm'] / calls
    return result


def render_report(summary):
    lines = [
        '# 项目框架与固定5000次NSGA-II调用比较', '',
        f"日期：{datetime.now(timezone.utc).date().isoformat()}。每个目标、每种NSGA-II方法的唯一物理调用上限固定为5000次；首次命中立即停止。框架调用与NSGA-II调用均不含F复算。", '',
        '| target | 框架调用 | 全域NSGA-II | 相同局部域NSGA-II |',
        '|---|---:|---:|---:|',
    ]
    grouped = {}
    for row in summary['rows']:
        grouped.setdefault(row['target_index'], {})[row['arm']] = row
    def value(row):
        return str(row['physical_calls']) if row['hit'] else f">{row['physical_calls']}（未命中）"
    for index in sorted(grouped):
        global_row = grouped[index]['global_cold_nsga2']
        local_row = grouped[index]['route_informed_local_nsga2']
        lines.append(f"| {index} | {global_row['project_discovery_calls']} | {value(global_row)} | {value(local_row)} |")
    lines.extend(['', '## 汇总', ''])
    for arm, label in [('global_cold_nsga2', '纯全域cold-start NSGA-II'),
                       ('route_informed_local_nsga2', '相同gate局部域NSGA-II')]:
        row = summary['aggregate'][arm]
        lines.append(f"- {label}：{row['hits']}/{row['tasks']}命中，实际物理调用{row['total_physics_calls']}次；双方命中的{row['both_hit_tasks']}个任务中，NSGA-II调用{row['both_hit_nsga2_calls']}次，框架调用{row['both_hit_project_calls']}次。")
    lines.extend(['',
        '全域基线不使用生成器、代理、gate、D-Flow或梯度增强。局部域基线仅借用冻结gate专家、0.20支持域及固定整数变量，不使用生成网络、代理排序、D-Flow或梯度增强。', '',
        '未命中结果是5000次预算下的右删失结果，不能把预算耗尽次数称为达到命中的调用数。任务来自开发困难集，不能外推为未知目标总体成功率。', '',
        f"证据：`{summary['output']}/Summary.json`、协议锁、完整清单及逐任务账本。", ''
    ])
    return '\n'.join(lines)


def run(*, resume=False, verify_only=False):
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    require(config['fixed_physics_call_budget_per_target_per_arm'] == 5000, 'Budget must be exactly 5000')
    data = g3.preflight(ROOT)
    project_root = within(ROOT, config['project_source'])
    output = within(ROOT, config['output'])
    identity = {
        'config': config, 'config_sha256': file_hash(ROOT / CONFIG),
        'runner_sha256': file_hash(Path(__file__)),
        'base_runner_sha256': file_hash(ROOT / 'scripts/run_geometry19_framework_vs_nsga2_10x_local.py'),
        'project_summary_sha256': file_hash(project_root / 'Summary.json'),
        'route_complete_manifest_sha256': file_hash(within(ROOT, config['route_source']) / 'Complete_Manifest.json'),
        'runtime': data['context'].contract_mapping(),
    }
    if resume or verify_only:
        lock = json.loads((output / 'Protocol_Lock.json').read_text())
        require(all(lock[key] == value for key, value in identity.items()), 'Identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        lock = {**identity, 'created_at': datetime.now(timezone.utc).isoformat()}
        g3.seal_json(output / 'Protocol_Lock.json', lock)
    project = {row['target_index']: row for row in json.loads((project_root / 'Summary.json').read_text())['targets']}
    with (output / '.runner.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for index in config['targets']:
            adapted = task_config(config, project[index])
            for arm in config['arms']:
                directory = output / f'target{index:03d}' / arm
                if directory.exists():
                    require((directory / 'Manifest.json').is_file(), 'Partial arm exists; preserve and stop')
                    row = base.verify(directory)
                else:
                    require(not verify_only, 'Campaign incomplete')
                    directory.mkdir(parents=True)
                    row = base.run_arm(data, adapted, project[index], index, arm, directory)
                    require(row['budget'] == 5000, 'Adapted task budget changed')
                rows.append(row)
                print(json.dumps(row), flush=True)
        summary = {'status': 'N7_FRAMEWORK_VS_NSGA2_FIXED5000_COMPLETED', 'rows': rows,
                   'aggregate': {arm: base.aggregate(rows, arm) for arm in config['arms']},
                   'fixed_budget': 5000, 'output': config['output']}
        if (output / 'Summary.json').exists():
            require(json.loads((output / 'Summary.json').read_text()) == summary, 'Summary changed')
        else:
            require(not verify_only, 'Summary missing'); g3.seal_json(output / 'Summary.json', summary)
        report = within(ROOT, config['report']); report.parent.mkdir(parents=True, exist_ok=True)
        if not verify_only: g3.adapter._write_once(report, render_report(summary).encode())
        require(report.is_file(), 'Report missing')
        complete = {'files': {str(path.relative_to(output)): file_hash(path) for path in output.rglob('*')
                    if path.is_file() and path.name not in {'.runner.lock', 'Complete_Manifest.json'}},
                    'report': {config['report']: file_hash(report)}}
        if (output / 'Complete_Manifest.json').exists():
            require(json.loads((output / 'Complete_Manifest.json').read_text()) == complete, 'Manifest changed')
        else:
            require(not verify_only, 'Manifest missing'); g3.seal_json(output / 'Complete_Manifest.json', complete)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--resume', action='store_true'); parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(); print(json.dumps(run(resume=args.resume, verify_only=args.verify_only), indent=2))
