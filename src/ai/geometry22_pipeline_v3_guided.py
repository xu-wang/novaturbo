"""Versioned trained V3 B--F flow with sealed stage boundaries and strict resume."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
import yaml

from src.ai.geometry19_pipeline_v3 import (
    Geometry19V3PendingError, OUTPUT_NAMES, _load_target_v3, _rho,
)
from src.ai.v3_local_cfm import load as load_cfm, integrate
from src.ai.v3_guidance import guide, refine
from src.ai.geometry19_performance_gated_coordinates import hard_failure
from src.ai.v3_extended_gate_l6f import network
from src.ai.v3_target_routed_local_l6c import _candidate, _draws, _route, _score
from src.ai.v3_validation_integrity import dependency_hashes
from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_pipeline_v3_guided_development_v1.yaml"
PROFILE = "v3_guided_cfm_development"
VERSION = "geometry22_pipeline_v3_guided_development_v1"
STAGES = list("BCDEF")
F_SELECTION_POLICY = "feasible_then_diagnostic_domain_valid_then_other_by_rho_and_design_id"
LEGACY_F_SELECTION_POLICY = "best_physics_rho_per_target_and_arm_after_evaluation"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19V3PendingError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, data: Any) -> None:
    """Replace a single JSON record atomically; stage directories are sealed separately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source(root: Path) -> tuple[dict, dict, dict, list, dict, dict]:
    cfg = yaml.safe_load((root / CONFIG).read_text())
    require(cfg["version"] == VERSION and cfg["pipeline_profile"] == PROFILE
            and cfg["stages"] == STAGES and cfg["stage_budgets"] == {"C": 8, "D": 8, "E": 8}
            and cfg["route_arm"] == "gated_top2" and cfg["raw_candidates_per_target_per_arm"] == 256
            and cfg["relative_tolerances"] == {name: .05 for name in OUTPUT_NAMES}
            and cfg["formal_use_allowed"] is False and cfg["deployment_allowed"] is False
            and cfg["holdout_labels_parsed"] is False
            and cfg["top2_allocation"] == "equal_128_each_if_two_else_256_single"
            and cfg["local_generation"]["method"] == "target_center_conditional_local_cfm"
            and cfg["local_generation"]["integer_policy"] == "fixed_from_routed_center"
            and cfg["physical_gpu_index"] == 6
            and cfg["f_recompute"] == F_SELECTION_POLICY, "V3 routed config changed scope")
    require(cfg['guidance']=={'steps':8,'learning_rate':.03,'objective':'center_local_ridge_log_mse','source_prior_weight':.001}
            and cfg['refinement']['steps']==24 and cfg['refinement']['learning_rate']==.03
            and cfg['refinement']['trigger']=='no_diagnostic_domain_valid_software_hit_in_C', 'Guided optimization config changed scope')
    spec = cfg["sources"]
    assets = root / spec["l6f_path"]
    for filename, key in (("fit_manifest.json", "fit_sha256"), ("gate_selection.json", "gate_selection_sha256"),
                          ("local_forward_models.json", "local_forward_sha256"), ("normalization.json", "normalization_sha256"),
                          ("center_support.json", "center_support_sha256")):
        require(sha(assets / filename) == spec[key], "V3 routed asset changed: " + filename)
    require(sha(root / spec["diagnostic_profile_path"]) == spec["diagnostic_profile_sha256"], "V3 routed profile changed")
    require(sha(root / spec['cfm_manifest_path']) == spec['cfm_manifest_sha256'], 'CFM asset manifest changed')
    require(read(root / spec['cfm_manifest_path'])['files']['checkpoint.pt'] == spec['cfm_checkpoint_sha256'], 'CFM checkpoint identity changed')
    fit = read(assets / "fit_manifest.json")
    local = read(assets / "local_forward_models.json")
    support = read(assets / "center_support.json")["centers"]
    norm = read(assets / "normalization.json")
    require(len(support) == 55 and len({t["center_id"] for t in support}) == 55
            and fit["selected_neural_model_id"] == "gate_seed_44"
            and local["selected_model_id"] == fit["selected_local_forward_model_id"]
            and fit["new_validation_or_holdout_labels_parsed"] is False, "V3 routed model population changed")
    checkpoint = next(t for t in fit["neural_models"] if t["model_id"] == fit["selected_neural_model_id"])
    require(sha(assets / checkpoint["path"]) == checkpoint["sha256"], "V3 routed checkpoint changed")
    return cfg, fit, local, support, norm, checkpoint


def code_hashes(root: Path) -> dict:
    # Preserve the L6h lock implementation. Add all pipeline-specific consumers.
    paths = set(dependency_hashes(root)) | {
        "src/ai/geometry22_pipeline_v3_guided.py", "src/ai/geometry19_pipeline_v3.py",
        "src/ai/v3_development_assets.py", "src/ai/geometry19_pipeline_dispatch.py",
        "scripts/run_wp19_23_pipeline.py", "src/ai/v3_local_cfm.py", "src/ai/v3_guidance.py",
    }
    return {name: sha(root / name) for name in sorted(paths)}


def generate(root: Path, target: dict, cfg: dict, local: dict, support: list, norm: dict, checkpoint: dict) -> tuple[list, list, dict]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "6" and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "V3 routed inference requires physical GPU 6 (CUDA_VISIBLE_DEVICES=6)")
    device = torch.device("cuda:0")
    print(f"V3 route GPU: visible=1 device={torch.cuda.get_device_name(0)} physical=6", flush=True)
    model = network().to(device)
    payload = torch.load(root / cfg["sources"]["l6f_path"] / checkpoint["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    public = {"target_id": target["target_id"], "target_performance": {name: target[name] for name in OUTPUT_NAMES}}
    route = _route(public, norm, support, model, device)
    choices = route["arms"][cfg["route_arm"]]
    if not choices:
        return [], [], route
    require(len(choices) in (1, 2), "Invalid V3 Top-2 population")
    normalizer = Geometry22Normalizer()
    continuous = local["input_continuous_indices_zero_based"]
    by_center = {c["center_id"]: c for c in support}
    flow,payload,verified = load_cfm(root,device)
    require(payload['class_ids'] == [c['center_id'] for c in support]
            and payload['continuous_indices'] == continuous, 'CFM routed chart identity changed')
    target_condition=(np.log([target[n] for n in OUTPUT_NAMES])-np.asarray(norm['log_mean']))/np.asarray(norm['log_std'])
    raw = []
    for center in choices:
        count=256//len(choices)
        seed=int(hashlib.sha256((target['target_id']+center+cfg['local_generation']['seed_salt']).encode()).hexdigest()[:8],16)
        generator=torch.Generator(device=device).manual_seed(seed)
        noise=torch.randn((count,19),generator=generator,device=device)*payload['config']['noise_std']
        condition=torch.tensor(target_condition,dtype=torch.float32,device=device).expand(count,-1)
        indices=torch.full((count,),payload['class_ids'].index(center),dtype=torch.long,device=device)
        asset=local['models'][local['selected_model_id']][center]
        coef=torch.tensor(asset['coefficients'],dtype=torch.float32,device=device)
        intercept=torch.tensor(asset['intercept_log'],dtype=torch.float32,device=device)
        goal=torch.tensor([target[n] for n in OUTPUT_NAMES],dtype=torch.float32,device=device)
        generated,history=guide(flow,noise,condition,indices,coef,intercept,goal,
                                steps=cfg['guidance']['steps'],lr=cfg['guidance']['learning_rate'])
        route.setdefault('dflow_history',{})[center]=history
        residuals=generated.cpu().numpy()
        for residual in residuals:
            row=decode_candidate(public,len(raw),by_center[center],residual,normalizer,continuous,cfg)
            row['generator_checkpoint_sha256']=verified['checkpoint_sha256']
            raw.append(row)
    _score(raw, public, local, continuous, by_center, device)
    ranked = sorted((r for r in raw if r["status"] == "geometric_candidate"),
                    key=lambda r: (r["predicted_rho"], r["design_id"], r["raw_slot"]))
    seen, selected = set(), []
    for row in ranked:
        if row["design_id"] in seen:
            continue
        seen.add(row["design_id"])
        selected.append({**row, "rank": len(selected)})
        if len(selected) == sum(cfg["stage_budgets"].values()):
            break
    return selected, raw, route


def decode_candidate(public,slot,center,residual,normalizer,continuous,cfg):
    row={'target_id':public['target_id'],'arm':cfg['route_arm'],'raw_slot':slot,
         'route_center_id':center['center_id'],'generator':'v3_local_cfm_v1'}
    residual=np.asarray(residual,dtype=np.float64)
    radius=float(np.linalg.norm(residual)*.08)
    row['normalized_radius']=radius if np.isfinite(radius) else None
    if residual.shape!=(19,) or not np.isfinite(residual).all() or radius>.08+1e-10:
        return row | {'status':'outside_local_support'}
    normalized=np.asarray(center['center_normalized_design'],dtype=np.float64).copy()
    normalized[continuous]+=.08*residual
    if np.any(normalized < -1) or np.any(normalized > 1):
        return row | {'status':'outside_schema'}
    try:
        design=normalizer.inverse(normalized.tolist())
    except ValueError:
        return row | {'status':'invalid_22d_design'}
    require(all(design.as_mapping()[n]==v for n,v in center['integer_state'].items()),'CFM integer state changed')
    actual=math.dist([normalizer.transform(design)[i] for i in continuous],
                     [center['center_normalized_design'][i] for i in continuous])
    require(math.isclose(actual,radius,abs_tol=1e-10,rel_tol=1e-8),'CFM serialized support changed')
    failure=hard_failure(design.as_mapping(),list(normalizer.fields),.2)
    if failure is not None:
        return row | {'status':'geometry_rejected:'+str(failure)}
    return row | {'status':'geometric_candidate','design_id':design.design_id(),'design':design.as_mapping(),
                  'normalized_design':list(normalizer.transform(design)),
                  'design_schema_hash':cfg['identity']['design_schema_hash'],'physics_bundle_hash':cfg['identity']['physics_bundle_hash']}


def evaluate_design(row: dict, runtime: Any, profile: Any, target: dict, cfg: dict) -> dict:
    design = Geometry22Design.from_mapping(row["design"])
    result = runtime.evaluate(design)
    engine = result.engine_result
    diagnostic = assess_constraint_profile(engine, profile, conservation_tolerances=runtime.base_runtime.conservation_tolerances).as_mapping()
    performance = {n: float(engine.performance[n]) for n in OUTPUT_NAMES} if engine.solver.converged else None
    if performance is not None and any(not math.isfinite(v) or v <= 0 for v in performance.values()):
        performance = None
    rho = _rho(performance, target, cfg["relative_tolerances"])
    return {"design_id": result.design_id, "design": design.as_mapping(), "route_center_id": row["route_center_id"],
            "design_schema_hash": result.design_schema_hash, "physics_bundle_hash": result.physics_bundle_hash,
            "solver_converged": bool(engine.solver.converged), "solver_failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
            "diagnostic_domain_valid": bool(diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"]),
            "feasible": bool(engine.feasible), "performance": performance, "rho": rho,
            "software_hit_5pct": bool(rho is not None and rho <= 1),
            "formal_use_allowed": False, "deployment_allowed": False}


def state_files(output: Path) -> dict:
    files = {}
    for p in sorted(output.rglob("*")):
        require(p.resolve().is_relative_to(output.resolve()), "V3 artifact path escapes run")
        if p.is_file() and not (p.parent == output and p.name in {"Run_State.json", ".runner.lock", "Complete_Manifest.json"}):
            files[str(p.relative_to(output))] = sha(p)
    return files


def verify_state(output: Path, identity: dict) -> dict:
    state = read(output / "Run_State.json")
    require(read(output / "Protocol_Lock.json")["identity"] == identity, "V3 routed identity changed")
    require(state["status"] in {"RUNNING", "COMPLETE"}, "Failed V3 routed run exists; preserve and stop")
    stages = state["completed_stages"]
    require(stages == STAGES[:len(stages)], "V3 routed stage order changed")
    require({p.name for p in output.iterdir() if p.is_dir()} == set(stages), "Unsealed V3 stage exists; preserve and stop")
    require(state["files"] == state_files(output), "V3 routed artifacts changed or unsealed stage exists; preserve and stop")
    require(read(output / "target.json") == identity["target"], "V3 routed target changed")
    return state


def best_result(evaluated: list, policy: str = F_SELECTION_POLICY) -> dict | None:
    require(policy in {F_SELECTION_POLICY, LEGACY_F_SELECTION_POLICY}, "Unknown guided F selection policy")
    usable = [r for r in evaluated if r["rho"] is not None]
    feasible = [r for r in usable if r["feasible"]]
    # Archived protocols retain their original ranking; new runs prefer valid diagnostics.
    diagnostic = [r for r in usable if r["diagnostic_domain_valid"]] if policy == F_SELECTION_POLICY else []
    return min(feasible or diagnostic or usable, key=lambda r: (r["rho"], r["design_id"])) if usable else None


def verify_relationships(output: Path, identity: dict, cfg: dict, stages: list) -> None:
    if "B" not in stages:
        return
    selected = read(output / "B/candidates.json")
    raw = read(output / "B/raw_candidates.json")
    route = read(output / "B/route.json")
    b = read(output / "B/summary.json")
    require(route["target_id"] == identity["target"]["target_id"], "V3 route target changed")
    require(route["status"] in {"routable", "abstained_no_target_support"}, "V3 route status changed")
    require(len(raw) == (256 if route["status"] == "routable" else 0), "V3 raw budget changed")
    require(len(selected) <= 24 and len({r["design_id"] for r in selected}) == len(selected), "V3 selected budget changed")
    valid = sorted((r for r in raw if r["status"] == "geometric_candidate"), key=lambda r: (r["predicted_rho"], r["design_id"], r["raw_slot"]))
    expected, seen = [], set()
    for r in valid:
        if r["design_id"] not in seen:
            seen.add(r["design_id"])
            expected.append({**r, "rank": len(expected)})
            if len(expected) == 24:
                break
    require(selected == expected and b["count"] == len(selected) and b["raw_slots"] == len(raw), "V3 B ranked selection changed")
    require(all(r.get('generator')=='v3_local_cfm_v1' for r in raw), 'V3 CFM candidate provenance changed')
    generator=identity['generator_contract']
    anchors={c['center_id']:c for c in generator['centers']}
    continuous=generator['continuous_indices']
    contract = identity["runtime_contract"]
    for r in selected:
        require(Geometry22Design.from_mapping(r["design"]).design_id() == r["design_id"], "V3 B design identity changed")
        require(all(r[k] == contract[k] for k in ("design_schema_hash", "physics_bundle_hash")), "V3 B backend identity changed")
        anchor=anchors[r['route_center_id']]
        actual=math.dist([r['normalized_design'][i] for i in continuous],[anchor['center_normalized_design'][i] for i in continuous])
        require(actual<=.08+1e-10 and math.isclose(actual,r['normalized_radius'],abs_tol=1e-10,rel_tol=1e-8) and all(r['design'][n]==v for n,v in anchor['integer_state'].items()), 'CFM archived chart changed')
        require(r['normalized_radius']<=.08+1e-10 and r['generator_checkpoint_sha256']==cfg['sources']['cfm_checkpoint_sha256'], 'V3 CFM support/asset changed')
        require(r["route_center_id"] in route["arms"][cfg["route_arm"]], "V3 selected center changed")
        require(r["target_id"] == identity["target"]["target_id"] and r["arm"] == cfg["route_arm"], "V3 candidate target changed")
    evaluated = []
    cursor = 0
    for stage in "CDE":
        if stage not in stages:
            break
        group = read(output / stage / "evaluations.json")
        expected_designs = read(output / stage / 'candidates.json')
        require(len(expected_designs)<=cfg['stage_budgets'][stage], 'V3 guided stage budget exceeded')
        should_run = stage=='C' or not any(diagnostic_hit(r) for r in evaluated)
        require(should_run or not expected_designs, 'Guided stage ran after diagnostic hit')
        if stage=='C':require(expected_designs==selected[:8], 'Guided C selection changed')
        if stage=='E':require(expected_designs==(selected[8:16] if should_run else []), 'Guided E fallback selection changed')
        for chosen in expected_designs:
            require(Geometry22Design.from_mapping(chosen['design']).design_id()==chosen['design_id'], 'Guided candidate identity changed')
            anchor=anchors[chosen['route_center_id']]
            z=Geometry22Normalizer().transform(Geometry22Design.from_mapping(chosen['design']))
            require(math.dist([z[i] for i in continuous],[anchor['center_normalized_design'][i] for i in continuous])<=.08+1e-10
                    and all(chosen['design'][n]==v for n,v in anchor['integer_state'].items()), 'Refinement left local support')
            if stage=='D':require(chosen['parent_design_id'] in {r['design_id'] for r in selected[:8]}, 'Refinement parent changed')
        require(len(group) == len(expected_designs), "V3 stage budget changed")
        for result, chosen in zip(group, expected_designs):
            require(result["design"] == chosen["design"] and result["design_id"] == chosen["design_id"]
                    and result["route_center_id"] == chosen["route_center_id"], "V3 evaluation design changed")
            require(all(result[k] == contract[k] for k in ("design_schema_hash", "physics_bundle_hash")), "V3 evaluation backend changed")
            expected_rho = _rho(result["performance"], identity["target"], cfg["relative_tolerances"])
            require(result["rho"] == expected_rho and result["software_hit_5pct"] == (expected_rho is not None and expected_rho <= 1), "V3 evaluation rho changed")
            require(result["formal_use_allowed"] is False and result["deployment_allowed"] is False, "V3 evaluation use changed")
        require(read(output / stage / "summary.json") == evaluation_summary(group), "V3 stage summary changed")
        evaluated.extend(group)
    if "F" in stages:
        best = best_result(evaluated, cfg["f_recompute"])
        f = read(output / "F/summary.json")
        require(f["independent_recompute"] == best, "V3 F independent recompute changed")
        require(f == final_summary(best), "V3 F terminal changed")
        if (output / "Pipeline_Summary.json").exists():
            summary = read(output / "Pipeline_Summary.json")
            require(summary == pipeline_summary(output, identity, selected, evaluated, f, route), "V3 pipeline summary changed")


def diagnostic_hit(row):
    return row['diagnostic_domain_valid'] and row['software_hit_5pct']


def refine_candidates(seeds,local,support,target,cfg):
    if not seeds:return [],[]
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='6' and torch.cuda.is_available() and torch.cuda.device_count()==1,
            'Continuous optimization requires physical GPU 6')
    device=torch.device('cuda:0');normalizer=Geometry22Normalizer()
    continuous=local['input_continuous_indices_zero_based'];by_center={c['center_id']:c for c in support}
    public={'target_id':target['target_id']};rows=[];histories=[]
    for seed in seeds:
        center=by_center[seed['route_center_id']]
        anchor=torch.tensor(center['center_normalized_design'],dtype=torch.float32,device=device)
        values=torch.tensor(seed['normalized_design'],dtype=torch.float32,device=device)
        residual=((values[continuous]-anchor[continuous])/.08)[None]
        asset=local['models'][local['selected_model_id']][center['center_id']]
        coef=torch.tensor(asset['coefficients'],dtype=torch.float32,device=device)
        intercept=torch.tensor(asset['intercept_log'],dtype=torch.float32,device=device)
        goal=torch.tensor([target[n] for n in OUTPUT_NAMES],dtype=torch.float32,device=device)
        refined,history=refine(residual,anchor,continuous,coef,intercept,goal,
                               steps=cfg['refinement']['steps'],lr=cfg['refinement']['learning_rate'])
        # Serialize within the declared float32 projection precision, then screen exactly.
        vector=refined[0].cpu().numpy().astype(np.float64)
        vector*=min(1.,(1.-1e-6)/max(float(np.linalg.norm(vector)),1e-12))
        row=decode_candidate(public,seed['raw_slot'],center,vector,normalizer,continuous,cfg)
        histories.append({'parent_design_id':seed['design_id'],'history':history,'decode_status':row['status']})
        if row['status']=='geometric_candidate':
            row.update(parent_design_id=seed['design_id'],generator_checkpoint_sha256=seed['generator_checkpoint_sha256'])
            if row['design_id'] not in {r['design_id'] for r in rows}:rows.append(row)
    _score(rows,{'target_id':target['target_id'],'target_performance':{n:target[n] for n in OUTPUT_NAMES}},local,continuous,by_center,device)
    return rows,histories


def evaluation_summary(group: list) -> dict:
    return {"evaluated_count": len(group), "solver_converged": sum(r["solver_converged"] for r in group),
            "diagnostic_domain_valid": sum(r["diagnostic_domain_valid"] for r in group),
            "feasible_count": sum(r["feasible"] for r in group), "software_hit_count": sum(r["software_hit_5pct"] for r in group)}


def final_summary(best: dict | None) -> dict:
    delivered = bool(best is not None and best["feasible"] and best["software_hit_5pct"])
    return {"status": "PASS" if delivered else "NO_HIT" if best else "NO_CANDIDATES",
            "selected_design_id": best["design_id"] if best else None, "independent_recompute": best,
            "software_hit_5pct": bool(best is not None and best["software_hit_5pct"])}


def pipeline_summary(output: Path, identity: dict, selected: list, evaluated: list, f: dict, route: dict) -> dict:
    return {"status": "V3_ROUTED_DEVELOPMENT_COMPLETED" if f["status"] == "PASS" else "V3_ROUTED_DEVELOPMENT_NO_DELIVERED_HIT",
            "model": "model V3", "pipeline_profile": PROFILE, "target_id": identity["target"]["target_id"],
            "terminal_source_stage": "F", "delivery_status": f["status"],
            "delivered_design_ids": [f["selected_design_id"]] if f["status"] == "PASS" else [],
            "route_status": route["status"], "eligible_center_count": route["eligible_center_count"],
            "software_hit_5pct": any(r["software_hit_5pct"] for r in evaluated),
            "candidate_count": len(selected), "physics_evaluation_count": len(evaluated) + (f["independent_recompute"] is not None),
            "diagnostic_domain_valid_count": sum(r["diagnostic_domain_valid"] for r in evaluated),
            "feasible_count": sum(r["feasible"] for r in evaluated), "design_dimension": 22,
            "design_schema_hash": identity["runtime_contract"]["design_schema_hash"],
            "physics_bundle_hash": identity["runtime_contract"]["physics_bundle_hash"],
            "asset_status": "frozen_development_gate_local_cfm_and_ridge_unvalidated",
            "physics_validation_status": "not_completed", "formal_use_allowed": False, "deployment_allowed": False,
            "paths": {"runtime_root": str(output), "pipeline_manifest": str(output / "Complete_Manifest.json")}}


def seal_stage(output: Path, stage: str, records: dict, state: dict) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".stage_", dir=output))
    try:
        for name, value in records.items():
            write(temporary / name, value)
        os.rename(temporary, output / stage)
        state["completed_stages"].append(stage)
        state["files"] = state_files(output)
        write(output / "Run_State.json", state)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@contextmanager
def run_lock(output: Path):
    with (output / ".runner.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Geometry19V3PendingError("V3 routed run is already active") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_v3_routed_pipeline(repository_root: str | Path, *, target_path=None, target_values=None,
                           target_type="ID", expected_status="UNKNOWN", output_root=None,
                           resume=False, device="cuda:0", refinement_device=None,
                           pipeline_profile=PROFILE, **kwargs) -> dict:
    require(pipeline_profile == PROFILE and device == "cuda:0" and refinement_device is None,
            "V3 routed profile requires --device cuda:0 and no refinement device")
    allowed = {"model_variant", "forward_model_tag", "sampling_mode", "storage_root", "restart_from_stage", "pipeline_context", "export_mesh"}
    require(set(kwargs) <= allowed and all(v is None or v is False for v in kwargs.values()), "Unsupported V3 routed option")
    root = Path(repository_root).resolve()
    cfg, fit, local, support, norm, checkpoint = source(root)
    target = _load_target_v3(target_path, target_values, target_type, expected_status)
    runtime = build_v3_development_physics_runtime(root)
    require(all(runtime.contract_mapping()[k] == cfg["identity"][k] for k in cfg["identity"]), "V3 routed runtime changed")
    identity = {"version": VERSION, "model": "model V3", "pipeline_profile": PROFILE, "protocol": cfg,
                "target": target, "config_sha256": sha(root / CONFIG), "source_code_sha256": code_hashes(root),
                "runtime_contract": runtime.contract_mapping(), "sources": cfg["sources"],
                "gate_checkpoint_sha256": checkpoint["sha256"], "local_model_id": local["selected_model_id"],
                "generator_contract": {"checkpoint_sha256":cfg['sources']['cfm_checkpoint_sha256'],
                                       "centers":support,"continuous_indices":local['input_continuous_indices_zero_based']},
                "inference_device": "cuda:0", "physical_gpu_index": 6, "physics_device": "cpu"}
    output = Path(output_root).resolve() if output_root else root / cfg["default_output_root"] / target["target_id"]
    if resume:
        require(output.is_dir(), "No V3 routed run to resume")
    else:
        require(not output.exists(), "Refusing to overwrite V3 routed run")
        output.mkdir(parents=True)
    with run_lock(output):
        if resume:
            state = verify_state(output, identity)
            verify_relationships(output, identity, cfg, state["completed_stages"])
            if state["status"] == "COMPLETE":
                manifest = read(output / "Complete_Manifest.json")
                require(manifest == {"manifest_version": VERSION, "stages": STAGES, "files": state["files"],
                                     "run_state_sha256": sha(output / "Run_State.json")}, "V3 routed complete manifest changed")
                return read(output / "Pipeline_Summary.json")
            require(not (output / "Complete_Manifest.json").exists(), "Unexpected V3 complete manifest")
        else:
            write(output / "Protocol_Lock.json", {"identity": identity})
            write(output / "target.json", target)
            state = {"status": "RUNNING", "completed_stages": [], "files": state_files(output)}
            write(output / "Run_State.json", state)
        try:
            if "B" not in state["completed_stages"]:
                selected, raw, route = generate(root, target, cfg, local, support, norm, checkpoint)
                seal_stage(output, "B", {"candidates.json": selected, "raw_candidates.json": raw, "route.json": route,
                           "summary.json": {"count": len(selected), "raw_slots": len(raw), "route_status": route["status"]}}, state)
            selected = read(output / "B/candidates.json")
            profile = ConstraintProfile.from_yaml(root / cfg["sources"]["diagnostic_profile_path"])
            evaluated = []
            for stage in 'CDE':
                if stage not in state['completed_stages']:
                    hit=any(diagnostic_hit(r) for r in evaluated)
                    if stage=='C':group=selected[:8];history=[]
                    elif hit:group=[];history=[]
                    elif stage=='D' and selected:group,history=refine_candidates(selected[:8],local,support,target,cfg)
                    elif stage=='D':group=[];history=[]
                    else:group=selected[8:16];history=[]
                    results=[]
                    for row in group:
                        result=evaluate_design(row,runtime,profile,target,cfg)
                        results.append(result)
                        # Fixed C window evaluates all eight; later stages retain their bound.
                    seal_stage(output,stage,{'candidates.json':group,'evaluations.json':results,
                               'optimization_history.json':history,'summary.json':evaluation_summary(results)},state)
                evaluated.extend(read(output / stage / 'evaluations.json'))
            if "F" not in state["completed_stages"]:
                best = best_result(evaluated, cfg["f_recompute"])
                if best is not None:
                    fresh = build_v3_development_physics_runtime(root)
                    require(fresh.contract_mapping() == runtime.contract_mapping(), "V3 F runtime identity changed")
                    repeated = evaluate_design(best, fresh, profile, target, cfg)
                    require(repeated == best, "V3 F independent recompute differs")
                f = final_summary(best)
                seal_stage(output, "F", {"summary.json": f}, state)
            route = read(output / "B/route.json")
            summary = pipeline_summary(output, identity, selected, evaluated, read(output / "F/summary.json"), route)
            write(output / "Pipeline_Summary.json", summary)
            state.update(status="COMPLETE", files=state_files(output))
            write(output / "Run_State.json", state)
            write(output / "Complete_Manifest.json", {"manifest_version": VERSION, "stages": STAGES,
                  "files": state["files"], "run_state_sha256": sha(output / "Run_State.json")})
            verify_state(output, identity)
            verify_relationships(output, identity, cfg, STAGES)
            return summary
        except Exception as exc:
            write(output / "Failure_Summary.json", {"status": "V3_ROUTED_DEVELOPMENT_FAILED", "model": "model V3",
                  "pipeline_profile": PROFILE, "error_type": type(exc).__name__, "error": str(exc),
                  "last_completed_stage": state["completed_stages"][-1] if state["completed_stages"] else None,
                  "formal_use_allowed": False, "deployment_allowed": False})
            state.update(status="FAILED", files=state_files(output))
            write(output / "Run_State.json", state)
            write(output / "Failure_Manifest.json", {"status": "V3_ROUTED_DEVELOPMENT_FAILED", "files": state["files"]})
            raise Geometry19V3PendingError(f"V3 routed run failed; see {output / 'Failure_Summary.json'}") from exc
