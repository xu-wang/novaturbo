"""Sealed rank-128 extension with the original G4 top-8 prefix preserved."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import yaml

from src.ai import geometry19_performance_gated_g3 as g3
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_far_precision_v1 import build_candidate
from src.physics.geometry19_v2_2_dataset import _evaluate_row

CONFIG = 'config/experiments/geometry19_performance_gated_g4_rank128_v1.yaml'
NAMES = ('thrust_N', 'tsfc_kg_N_s', 'mass_kg')
CONTEXT = None


def write_rows(path, rows):
    payload = b''.join(canonical(r) + b'\n' for r in rows)
    if path.exists():
        require(path.read_bytes() == payload, f'Sealed rows changed: {path}')
    else:
        g3.adapter._write_once(path, payload)


def task_selection(raw, prediction, old_selection):
    require(len(raw) == len(prediction) == 256 and len(old_selection) == 8, 'G3 population mismatch')
    require([r['raw_slot_id'] for r in raw] == list(range(256)), 'Raw order changed')
    require([r['raw_slot_id'] for r in prediction] == list(range(256)), 'Prediction order changed')
    ranked = sorted((r for r in prediction if r['rank'] is not None), key=lambda r: r['rank'])
    require([r['rank'] for r in ranked] == list(range(1, len(ranked)+1)), 'Noncontiguous ranking')
    require(ranked == sorted(ranked, key=lambda r: (r['predicted_rho'], r['design_hash'])), 'Score ordering changed')
    require(len({r['design_hash'] for r in ranked}) == len(ranked), 'Duplicate ranked geometry')
    common = {k: raw[0][k] for k in ('target_id', 'target_index', 'arm', 'training_seed', 'generation_seed')}
    selected = []
    for rank in range(1, 129):
        row = dict(common, selection_rank=rank, missing=rank > len(ranked))
        if row['missing']:
            row.update(raw_slot_id=None, geometry19=None, design_hash=None, support_root_id=None,
                       predicted_rho=None, failure_reason='insufficient_valid_unique_candidates')
        else:
            p = ranked[rank-1]
            r = raw[p['raw_slot_id']]
            require(p['valid'] and r['valid'] and r['support_valid'] and r['geometry_hard_valid'], 'Invalid selection')
            require(r['design_hash'] == p['design_hash'], 'Raw/prediction mismatch')
            row.update({k: r[k] for k in ('raw_slot_id', 'geometry19', 'design_hash', 'support_root_id')})
            row['predicted_rho'] = p['predicted_rho']
        selected.append(row)
    require(selected[:8] == old_selection, 'Original top-8 changed')
    return selected


def preflight(root):
    config = yaml.safe_load((root / CONFIG).read_text())
    require(config['population'] == dict(tasks=1188, ranks=128, slots=152064, original_G4_prefix=8), 'Budget changed')
    data = g3.preflight(root)
    base = within(root, config['g3_root'])
    require(file_hash(base/'Complete_Manifest.json') == config['g3_complete_sha256'], 'G3 manifest changed')
    lock = json.loads((base/'Protocol_Lock.json').read_text())
    require(lock['code'] == data['code'] and lock['inputs'] == data['inputs'], 'G3 dependency mismatch')
    g3.verify_complete(base, lock)
    context = data['context']
    require(context.context_hash == config['physics']['context_sha256'] and
            context.physics_runtime.physics_bundle_hash == config['physics']['bundle_sha256'] and
            context.physics_runtime.model_version == config['physics']['model_version'], 'G4 runtime mismatch')
    return config, data, base


def prepare_plan(root, config, data, base):
    slots, unique = [], {}
    for task in g3.task_list(range(99)):
        raw = read_rows(base/'raw'/g3.task_key(task)/'Raw.jsonl')
        pred = read_rows(base/'ranked'/g3.task_key(task)/'Predictions.jsonl')
        old = read_rows(base/'ranked'/g3.task_key(task)/'Selection.jsonl')
        for row in task_selection(raw, pred, old):
            row['physics_index'] = None
            if not row['missing']:
                dh = row['design_hash']
                if dh not in unique:
                    unique[dh] = dict(physics_index=len(unique), design_hash=dh, geometry19=row['geometry19'])
                require(unique[dh]['geometry19'] == row['geometry19'], 'Hash collision')
                row['physics_index'] = unique[dh]['physics_index']
            slots.append(row)
    require(len(slots) == config['population']['slots'], 'Selection count changed')
    return slots, list(unique.values())


def identity(root, config, data):
    code = {p: file_hash(root/p) for p in (
        'src/ai/geometry19_performance_gated_g4.py', 'scripts/run_geometry19_performance_gated_g4.py',
        'tests/test_geometry19_performance_gated_g4.py', 'src/physics/geometry19_v2_2_dataset.py')}
    return dict(protocol_sha256=file_hash(root/CONFIG), code=code, g3_complete_sha256=config['g3_complete_sha256'],
                runtime_contract=data['context'].contract_mapping(), numpy_version=np.__version__)


def worker_init(root, expected_context):
    global CONTEXT
    root = Path(root)
    CONTEXT, _, _ = build_candidate(build_pipeline_context(root, pipeline_profile='v2_2_development'), root)
    require(CONTEXT.contract_mapping() == expected_context, 'Worker runtime differs')


def shard_paths(output, index):
    base = output/'physics'/f'part-{index:05d}'
    return base.with_suffix('.jsonl'), base.with_suffix('.calls.jsonl'), base.with_suffix('.seal.json')


def verify_shard(output, index, batch, lock_hash):
    path, journal, seal_path = shard_paths(output, index)
    existence = [p.exists() for p in (path, journal, seal_path)]
    require(sum(existence) in (0, 3), 'Orphan partial physics shard; preserve, no automatic replay')
    if not any(existence):
        return None
    seal = json.loads(seal_path.read_text())
    require(seal['identity'] == dict(lock_sha256=lock_hash, input_sha256=g3.digest(batch)), 'Shard identity differs')
    require(file_hash(path) == seal['results_sha256'] and file_hash(journal) == seal['journal_sha256'], 'Shard hash differs')
    rows = read_rows(path)
    require(len(rows) == seal['actual_calls'] == len(batch), 'Physics count differs')
    require(read_rows(journal) == [dict(physics_index=r['physics_index'], design_hash=r['design_hash'], event='call_started') for r in batch], 'Journal differs')
    for original, actual in zip(batch, rows):
        require(all(actual[k] == v for k, v in original.items()), 'Result input changed')
    return seal


def acquire_job(job):
    output, index, batch, lock_hash = job
    output = Path(output)
    old = verify_shard(output, index, batch, lock_hash)
    if old is not None:
        return index, old
    path, journal, seal_path = shard_paths(output, index)
    start = time.monotonic()
    with journal.open('xb') as calls, path.open('xb') as result_stream:
        for source in batch:
            calls.write(canonical(dict(physics_index=source['physics_index'], design_hash=source['design_hash'], event='call_started'))+b'\n')
            calls.flush()
            os.fsync(calls.fileno())
            # Existing physical evaluator; no target or source-root is passed to physics.
            result = _evaluate_row(dict(source), CONTEXT)
            require(result['pipeline_context_hash'] == CONTEXT.context_hash, 'Result runtime differs')
            result_stream.write(canonical(result)+b'\n')
            result_stream.flush()
        os.fsync(result_stream.fileno())
    seal = dict(identity=dict(lock_sha256=lock_hash, input_sha256=g3.digest(batch)), actual_calls=len(batch),
                elapsed_seconds=time.monotonic()-start, results_sha256=file_hash(path), journal_sha256=file_hash(journal),
                infrastructure_replays=0)
    g3.seal_json(seal_path, seal)
    return index, seal


def evaluate_jobs(root, output, unique, lock, workers, shard_size):
    jobs = [(str(output), i//shard_size, unique[i:i+shard_size], g3.digest(lock)) for i in range(0, len(unique), shard_size)]
    seals, pending_jobs = {}, []
    # All existing evidence must pass before any new physical call.
    expected = {p for _, i, _, _ in jobs for p in shard_paths(output, i)}
    require(set((output/'physics').iterdir()) <= expected, 'Unexpected physics files')
    for job in jobs:
        seal = verify_shard(output, job[1], job[2], job[3])
        if seal is None:
            pending_jobs.append(job)
        else:
            seals[job[1]] = seal
    started = time.monotonic()
    iterator = iter(pending_jobs)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=(str(root), lock['runtime_contract'])) as pool:
        pending = set()
        for _ in range(workers):
            job = next(iterator, None)
            if job is not None:
                pending.add(pool.submit(acquire_job, job))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index, seal = future.result()
                seals[index] = seal
                job = next(iterator, None)
                if job is not None:
                    pending.add(pool.submit(acquire_job, job))
            if len(seals) % 16 == 0 or not pending:
                print(json.dumps(dict(stage='physics', sealed_shards=len(seals), total_shards=len(jobs),
                                      calls=sum(s['actual_calls'] for s in seals.values()), elapsed_s=round(time.monotonic()-started,1))), flush=True)
    return [seals[i] for i in range(len(jobs))]


def score_slot(slot, physics, target, tolerances):
    row = {k: v for k, v in slot.items() if k != 'geometry19'}
    row.update(hit=False, rho=None, loss=1., solver_converged=False, numerically_valid=False,
               physics_domain_valid=False, target_domain_inside=False, physics_eligible=False)
    if slot['missing']:
        return row
    require(physics['design_hash'] == slot['design_hash'], 'Wrong physics lookup')
    assessment = physics['constraint_assessment']
    row.update(solver_converged=physics['solver']['converged'], numerically_valid=assessment['numerically_valid'],
               physics_domain_valid=assessment['physics_domain_valid'],
               target_domain_inside=(physics.get('target_domain_assessment') or {}).get('inside') is True)
    values = physics.get('performance') or {}
    finite = all(type(values.get(k)) in (int, float) and math.isfinite(values[k]) and values[k]>0 for k in NAMES)
    row['physics_eligible'] = finite and all(row[k] for k in ('solver_converged','numerically_valid','physics_domain_valid','target_domain_inside'))
    if finite:
        row['rho'] = max(abs(values[k]-target['performance'][k])/target['performance'][k]/tolerances[k] for k in NAMES)
        require(math.isfinite(row['rho']), 'Nonfinite rho')
    if row['physics_eligible']:
        row['loss'] = min(row['rho'], 10.)/10.
        row['hit'] = row['rho'] <= 1.
    return row


def task_metrics(rows, target):
    require(len(rows) == 128 and [r['selection_rank'] for r in rows] == list(range(1,129)), 'Rank denominator changed')
    hits = [int(r['hit']) for r in rows]
    cumulative = np.cumsum(hits).tolist()
    first = next((i+1 for i, h in enumerate(hits) if h), None)
    firstrow = rows[0]
    return dict(training_seed=firstrow['training_seed'], target_index=firstrow['target_index'], arm=firstrow['arm'],
                source_root_id=target['root_seed_id'], target_class=target['target_class'],
                first_hit_rank=first, cumulative_hit_count=cumulative, exact_rank_hits=hits,
                hit_count_128=cumulative[-1], candidate_hit_fraction_128=cumulative[-1]/128,
                at_least_half_hit=cumulative[-1]>=64, missing_count_128=sum(r['missing'] for r in rows),
                missing_count_8=sum(r['missing'] for r in rows[:8]), rank1_loss=firstrow['loss'],
                rank1_solver_converged=firstrow['solver_converged'], rank1_physics_domain_valid=firstrow['physics_domain_valid'],
                rank1_hit=firstrow['hit'])


def aggregate(tasks):
    count = len(tasks)
    return dict(tasks=count, hit_at_k=[sum(t['cumulative_hit_count'][k]>0 for t in tasks) for k in range(128)],
                exact_rank_hit_count=[sum(t['exact_rank_hits'][k] for t in tasks) for k in range(128)],
                cumulative_candidate_hits=[sum(t['cumulative_hit_count'][k] for t in tasks) for k in range(128)],
                hit_count_128=sum(t['hit_count_128'] for t in tasks),
                candidate_hit_fraction_128=sum(t['hit_count_128'] for t in tasks)/(128*count),
                tasks_with_at_least_64_hits=sum(t['at_least_half_hit'] for t in tasks),
                missing_count_128=sum(t['missing_count_128'] for t in tasks), missing_count_8=sum(t['missing_count_8'] for t in tasks))


def frozen_gates(tasks, data, g3summary):
    primary = [t for t in tasks if t['arm'] == g3.ARMS[0]]
    gates = data['parent']['evaluation']['gates_each_seed']
    by_seed = {}
    for seed in (42,43,44):
        part = [t for t in primary if t['training_seed'] == seed]
        roots = Counter(t['source_root_id'] for t in part if t['rank1_hit'])
        counts = dict(solver=sum(t['rank1_solver_converged'] for t in part),
                      physics_domain=sum(t['rank1_physics_domain_valid'] for t in part), hits=sum(t['rank1_hit'] for t in part))
        checks = dict(solver=counts['solver']>=gates['top1_solver_converged_of_99'],
                      physics_domain=counts['physics_domain']>=gates['top1_physics_domain_valid_of_99'],
                      hits=counts['hits']>=gates['top1_hits_of_99'],
                      each_root=all(roots[r]>=gates['top1_hits_each_root_of_3'] for r in data['decoders']['root_order']),
                      raw_support=g3summary['top1_raw_support_gate_each_seed'][str(seed)])
        by_seed[str(seed)] = dict(counts=counts, checks=checks, passed=all(checks.values()),
                                  root_hits={r: roots[r] for r in data['decoders']['root_order']})
    paired = data['parent']['evaluation']['paired_test']
    differences = []
    for root in data['decoders']['root_order']:
        a = [t['rank1_loss'] for t in primary if t['source_root_id']==root]
        b = [t['rank1_loss'] for t in tasks if t['arm']==g3.ARMS[3] and t['source_root_id']==root]
        require(len(a)==len(b)==9, 'Root pairing changed')
        differences.append(float(np.mean(b)-np.mean(a)))
    rng = np.random.default_rng(paired['bootstrap_seed'])
    boot = np.asarray(differences)[rng.integers(0,33,size=(paired['bootstrap_repetitions'],33))].mean(axis=1)
    low, high = np.quantile(boot, [.025,.975]).tolist()
    improved = sum(d>0 for d in differences)
    pair_result = dict(root_differences=dict(zip(data['decoders']['root_order'], differences)), mean=float(np.mean(differences)),
                       ci95=[low, high], improved_roots=improved, passed=low>0 and improved>=paired['minimum_improved_roots'])
    return dict(by_seed=by_seed, paired_rank1=pair_result,
                primary_physics_gates_and_paired_pass=all(v['passed'] for v in by_seed.values()) and pair_result['passed'])


def finalize(output, slots, unique, seals, data, config, base, lock):
    paths = [shard_paths(output, i)[0] for i in range(len(seals))]
    g3.concatenate(output, 'Physical_Evaluations.jsonl', paths)
    physics = read_rows(output/'Physical_Evaluations.jsonl')
    require(len(physics)==len(unique) and [r['physics_index'] for r in physics]==list(range(len(unique))), 'Physics population differs')
    g3.seal_json(output/'Acquisition_Manifest.json', dict(lock_sha256=g3.digest(lock), actual_calls=len(physics),
                  seals=seals, ledger_sha256=file_hash(output/'Physical_Evaluations.jsonl'), infrastructure_replays=0))
    scored, tasks = [], []
    tolerances = data['parent']['evaluation']['relative_tolerances']
    for start in range(0,len(slots),128):
        part = slots[start:start+128]
        target = data['targets'][part[0]['target_index']]
        rows = [score_slot(s, None if s['missing'] else physics[s['physics_index']], target, tolerances) for s in part]
        scored.extend(rows)
        tasks.append(task_metrics(rows,target))
    write_rows(output/'Ranked_Physics_Ledger.jsonl', scored)
    write_rows(output/'Original_G4_Top8_Ledger.jsonl', [r for r in scored if r['selection_rank']<=8])
    write_rows(output/'Task_Metrics.jsonl', tasks)
    by_arm = {a: aggregate([t for t in tasks if t['arm']==a]) for a in g3.ARMS}
    curves=[]
    for arm, values in by_arm.items():
        for k in range(1,129):
            curves.append(dict(arm=arm, rank=k, tasks=values['tasks'], hit_tasks=values['hit_at_k'][k-1],
                               cumulative_target_hit_rate=values['hit_at_k'][k-1]/values['tasks'],
                               exact_rank_candidate_hit_rate=values['exact_rank_hit_count'][k-1]/values['tasks'],
                               cumulative_candidate_hits=values['cumulative_candidate_hits'][k-1],
                               cumulative_candidate_hit_fraction=values['cumulative_candidate_hits'][k-1]/(values['tasks']*k)))
    write_rows(output/'Rank_Curves.jsonl', curves)
    summary = dict(status='G4_RANK128_PHYSICS_SEALED', tasks=len(tasks), slots=len(slots), actual_calls=len(physics),
                   selected_slots=sum(not s['missing'] for s in slots), missing_slots=sum(s['missing'] for s in slots),
                   exact_geometry_cache_reuses=sum(not s['missing'] for s in slots)-len(physics),
                   by_arm=by_arm, by_seed_arm={f'{seed}/{arm}':aggregate([t for t in tasks if t['training_seed']==seed and t['arm']==arm]) for seed in (42,43,44) for arm in g3.ARMS},
                   by_class_arm={f'{cls}/{arm}':aggregate([t for t in tasks if t['target_class']==cls and t['arm']==arm]) for cls in ('center','typical','outer') for arm in g3.ARMS},
                   noncenter_by_arm={a:aggregate([t for t in tasks if t['arm']==a and t['target_class']!='center']) for a in g3.ARMS},
                   original_gates=frozen_gates(tasks,data,json.loads((base/'G3_Summary.json').read_text())),
                   physics_context_hash=data['context'].context_hash, infrastructure_replays=0,
                   dflow=False, gradient_enhancement=False, nsga2=False, confirmation_content_read=False,
                   formal_use_allowed=False, deployment_allowed=False)
    g3.seal_json(output/'G4_Summary.json', summary)
    lines=['# G4 V2.2 physical evaluation: fixed top-128 extension','',
           f"Tasks: {len(tasks)}; slots: {len(slots)}; selected: {summary['selected_slots']}; missing: {summary['missing_slots']}.",
           f"Actual unique physics calls: {len(physics)}; exact-geometry cache reuses: {summary['exact_geometry_cache_reuses']}.",'',
           '| Arm | hit@1 | hit@8 | hit@32 | hit@64 | hit@128 | Candidate hits / 38,016 | Tasks with >=64 hits |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,v in by_arm.items():
        lines.append('| '+a+' | '+' | '.join(str(v['hit_at_k'][k-1])+'/297' for k in (1,8,32,64,128))+f" | {v['hit_count_128']} | {v['tasks_with_at_least_64_hits']} |")
    lines += ['', 'hit@k counts targets with at least one physical hit in the first k pre-ranked slots. It does not mean k candidates hit. Candidate hit counts and first-hit rank are reported separately. Missing slots remain failures. No early stopping after a hit.', '',
              f"Original rank-1 physics and root-paired gates pass: {summary['original_gates']['primary_physics_gates_and_paired_pass']}.",
              'The original top-8 ledger remains separate. Rank-128 is a user-authorized descriptive extension, not a replacement for frozen rank-1/top-8 acceptance. These are development targets, not independent confirmation.', '']
    g3.adapter._write_once(output/'G4_Report.md', '\n'.join(lines).encode())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(12,4))
    for a,v in by_arm.items():
        k=np.arange(1,129)
        axes[0].plot(k,np.asarray(v['hit_at_k'])/297,label=a)
        axes[1].plot(k,np.asarray(v['cumulative_candidate_hits'])/(297*k),label=a)
    axes[0].set_ylabel('Targets with >=1 hit / 297')
    axes[1].set_ylabel('Hits / fixed evaluated rank slots')
    for ax in axes:
        ax.set_xlabel('Surrogate rank cutoff k'); ax.set_ylim(0,1.02); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'Rank_Curves.png',dpi=160); plt.close(fig)
    files={str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in {'.runner.lock','Complete_Manifest.json'}}
    g3.seal_json(output/'Complete_Manifest.json', dict(lock_sha256=g3.digest(lock),files=files))
    return summary


def verify_complete(output, lock):
    manifest=json.loads((output/'Complete_Manifest.json').read_text())
    require(manifest['lock_sha256']==g3.digest(lock), 'Complete lock differs')
    for name, expected in manifest['files'].items():
        require(file_hash(within(output,name))==expected, f'Complete artifact changed: {name}')
    return json.loads((output/'G4_Summary.json').read_text())


def run(root, *, resume=False, verify_only=False):
    root=Path(root).resolve()
    config,data,base=preflight(root)
    output=within(root, config['output_root'])
    ident=identity(root,config,data)
    if resume or verify_only:
        lock=json.loads((output/'Protocol_Lock.json').read_text())
        require(all(lock[k]==v for k,v in ident.items()), 'Resume identity differs')
        if verify_only or (output/'Complete_Manifest.json').exists():
            return verify_complete(output,lock)
    else:
        output.mkdir(parents=True,exist_ok=False)
        lock=dict(ident,created_at=datetime.now(timezone.utc).isoformat())
        g3.seal_json(output/'Protocol_Lock.json',lock)
        (output/'physics').mkdir()
    with (output/'.runner.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        slots,unique=prepare_plan(root,config,data,base)
        write_rows(output/'Selection_Rank128.jsonl',slots)
        write_rows(output/'Unique_Physics_Plan.jsonl',unique)
        selection_manifest=dict(lock_sha256=g3.digest(lock), slots=len(slots), unique_calls=len(unique),
                                selection_sha256=file_hash(output/'Selection_Rank128.jsonl'),
                                physics_plan_sha256=file_hash(output/'Unique_Physics_Plan.jsonl'),
                                original_top8_identical=True)
        g3.seal_json(output/'Selection_Manifest.json',selection_manifest)
        print(json.dumps(dict(stage='selection_sealed',**selection_manifest)),flush=True)
        seals=evaluate_jobs(root,output,unique,lock,config['physics']['workers'],config['physics']['unique_calls_per_shard'])
        return finalize(output,slots,unique,seals,data,config,base,lock)
