"""Train-only transforms and immutable G1/G2 views. No torch or physics calls."""
from collections import Counter, defaultdict
import contextlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import yaml

from src.ai import geometry19_far_precision_relabel as migration
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within, OUTPUTS
from src.ai.geometry19_performance_gated_physics import check_row, write_json, write_once
from src.ai.geometry19_performance_gated_physics_protocol import eligibility_flags
from src.ai.geometry19_far_precision_diagnostic import digest

CONFIG = "config/experiments/geometry19_performance_gated_training_v1.yaml"


def fit_condition(train, roots):
    require(bool(train) and all(r["split"] == "train" and r["training_eligible"] for r in train), "Transform fit requires eligible train only")
    counts = Counter(r["root_seed_id"] for r in train)
    require(set(counts) == set(roots) and len(roots) == len(set(roots)), "Missing/duplicate training roots")
    logs = np.log(np.asarray([[r["performance"][k] for k in OUTPUTS] for r in train], dtype=np.float64))
    require(np.isfinite(logs).all(), "Invalid condition values")
    weights = np.asarray([1.0/(len(roots)*counts[r["root_seed_id"]]) for r in train], dtype=np.float64)
    mean = np.sum(weights[:,None]*logs,axis=0)
    variance = np.sum(weights[:,None]*(logs-mean)**2,axis=0)
    std = np.sqrt(variance)
    require(np.isfinite(std).all() and np.all(std>0), "Degenerate condition variance")
    return {"axes": OUTPUTS, "dimension": 3, "transform": "natural_log_then_zscore", "log_mean": mean.tolist(),
            "log_std": std.tolist(), "variance_ddof": 0, "fit_split": "train", "fit_rows": len(train),
            "root_count": len(roots), "per_root_fit_rows": dict(counts), "row_weight": "1/(root_count*root_train_rows)",
            "fit_design_hash_sequence_sha256": digest([r["design_hash"] for r in train])}


def transform_condition(performance, transform):
    require(set(performance) == set(OUTPUTS), "Condition accepts exactly three performance fields")
    values = np.asarray([performance[k] for k in OUTPUTS],dtype=np.float64)
    require(np.isfinite(values).all() and np.all(values>0), "Nonpositive/nonfinite performance")
    result = (np.log(values)-np.asarray(transform["log_mean"]))/np.asarray(transform["log_std"])
    require(result.shape == (3,) and np.isfinite(result).all(), "Invalid condition tensor")
    return result.tolist()


def make_decoders(roots, anchors, stable, variables):
    cont = [v for v in variables if v["dtype"] != "int64"]
    ints = [v for v in variables if v["dtype"] == "int64"]
    require(len(cont)==16 and len(ints)==3, "Geometry dimensions changed")
    lower = np.array([v["bounds"][0] for v in cont],dtype=float)
    span = np.array([v["bounds"][1]-v["bounds"][0] for v in cont],dtype=float)
    categories = {v["name"]: list(range(int(v["bounds"][0]),int(v["bounds"][1])+1)) for v in ints}
    decoders = []
    for index,root in enumerate(roots):
        anchor = anchors[root]["medoid_geometry19"]
        radius = stable[root]["support_radius_p95_normalized_l2"]
        require(radius>0 and np.isfinite(radius), "Invalid frozen root radius")
        center = ((np.array([anchor[v["name"]] for v in cont])-lower)/span).tolist()
        fixed = [anchor[v["name"]] for v in ints]
        require(all(type(x) is int and x in categories[v["name"]] for v,x in zip(ints,fixed)), "Invalid frozen integers")
        decoders.append({"root_seed_id":root,"root_index":index,"normalized_medoid_continuous":center,
                         "radius":radius,"integer_values":fixed,"anchor_design_hash":anchors[root]["medoid_design_hash"]})
    return {"root_order":roots,"continuous_axes":[v["name"] for v in cont],"integer_axes":[v["name"] for v in ints],
            "continuous_lower":lower.tolist(),"continuous_span":span.tolist(),"integer_categories":categories,
            "experts":decoders,"condition_dim":3,"decoder_parameters_are_not_conditions":True}


def project(rows, transform, decoders):
    result = []
    by_root = {d["root_seed_id"]:d for d in decoders["experts"]}
    counts = Counter(r["root_seed_id"] for r in rows)
    lower,span = np.array(decoders["continuous_lower"]),np.array(decoders["continuous_span"])
    for r in rows:
        d = by_root[r["root_seed_id"]]
        g = r["geometry19"]
        z = (np.array([g[k] for k in decoders["continuous_axes"]])-lower)/span
        u = (z-np.array(d["normalized_medoid_continuous"]))/d["radius"]
        integers = [g[k] for k in decoders["integer_axes"]]
        require(integers==d["integer_values"], "Integer state changed")
        require(np.isfinite(u).all() and np.linalg.norm(u)<=.20+1e-12 and np.all(z>=0) and np.all(z<=1), "Local geometry outside frozen support")
        decoded = lower+span*(np.array(d["normalized_medoid_continuous"])+d["radius"]*u)
        original = np.array([g[k] for k in decoders["continuous_axes"]])
        require(np.max(np.abs(decoded-original)/span)<=1e-12, "Decoder round trip failed")
        result.append({"design_hash":r["design_hash"],"pair_id":r["pair_id"],"split":r["split"],"band_index":r["band_index"],
                       "supervision":{"root_index":d["root_index"],"root_seed_id":r["root_seed_id"]},
                       "condition":transform_condition(r["performance"],transform),"local_state":u.tolist(),"pooled_state":z.tolist(),
                       "integer_class_indices":[decoders["integer_categories"][k].index(g[k]) for k in decoders["integer_axes"]],
                       "root_equal_row_weight":len(rows)/(len(by_root)*counts[r["root_seed_id"]])})
    return result


def prepare_views(rows, anchors, stable, variables):
    roots = sorted(anchors)
    require(len(roots)==33 and set(roots).issubset(stable), "Frozen root set changed")
    seen, pairs = set(),defaultdict(set)
    for r in rows:
        require(r["design_hash"] not in seen, "Duplicate design")
        seen.add(r["design_hash"])
        pairs[r["pair_id"]].add(r["split"])
        require(all(r[k]==v for k,v in eligibility_flags(r).items()), "Eligibility mismatch")
        require(r["split"]==("validation" if r["pair_index"]%5==0 else "train"), "Original split changed")
    require(all(len(v)==1 for v in pairs.values()), "Pair split leakage")
    train = [r for r in rows if r["training_eligible"]]
    validation = [r for r in rows if r["validation_eligible"]]
    excluded = [{"design_hash":r["design_hash"],"pair_id":r["pair_id"],"split":r["split"],"root_seed_id":r["root_seed_id"],
                 "reason":"not_physics_eligible","original_label_retained":True} for r in rows if not r["physics_eligible"]]
    require({r["pair_id"] for r in train}.isdisjoint({r["pair_id"] for r in validation}), "Train/validation pair overlap")
    transform = fit_condition(train,roots)
    decoders = make_decoders(roots,anchors,stable,variables)
    return project(train,transform,decoders),project(validation,transform,decoders),excluded,transform,decoders


def validation_noise(rows, seed):
    rng = np.random.Generator(np.random.PCG64(20260930+seed))
    noise = rng.standard_normal((len(rows),16))
    times = rng.random(len(rows))
    return [{"design_hash":r["design_hash"],"epsilon":noise[j].tolist(),"time":float(times[j])} for j,r in enumerate(rows)]


def verified_sources(root, config):
    paths = {}
    for role,spec in config["inputs"].items():
        paths[role]=within(root,spec["path"])
        require(file_hash(paths[role])==spec["sha256"],f"Training input changed: {role}")
    upstream=migration.load(root)
    migration.verify_prepared(root/migration.OUTPUT,upstream)
    def forbidden(*args,**kwargs):
        raise RuntimeError("Training preparation cannot evaluate physics")
    # Both execution summaries must already exist; only verified completed resume is permitted.
    output=root/migration.OUTPUT
    require((output/"Regression_Summary.json").is_file() and (output/"Relabel_Summary.json").is_file(), "Missing completed migration")
    with contextlib.redirect_stdout(io.StringIO()):
        summary=migration.execute_stages(output,upstream,"relabel",resume=True,candidate_eval=forbidden,parent_eval=forbidden)
    require(summary["status"]=="RELABEL_PASS", "Training labels have not passed")
    rows=read_rows(paths["labels"])
    for source,row in zip(upstream.rows,rows):
        check_row(source,row,upstream.candidate)
    require(len(rows)==len(upstream.rows)==16896, "Wrong label population")
    require(upstream.candidate.context_hash==config["physics"]["context_sha256"] and
            upstream.candidate.physics_runtime.physics_bundle_hash==config["physics"]["bundle_sha256"] and
            upstream.candidate.physics_runtime.model_version==config["physics"]["model_version"], "Wrong label physics version")
    parent=yaml.safe_load(paths["parent_protocol"].read_text())
    allowed={}
    for name in ("anchor_registry","stable_root_registry","schema"):
        spec=parent["inputs"][name]
        p=within(root,spec["path"])
        require(file_hash(p)==spec["sha256"],f"Decoder source changed: {name}")
        allowed[name]=p
    anchors={r["root_seed_id"]:r for r in read_rows(allowed["anchor_registry"])}
    stable={r["root_seed_id"]:r for r in read_rows(allowed["stable_root_registry"])}
    variables=sorted(yaml.safe_load(allowed["schema"].read_text())["design_variables"],key=lambda v:v["index"])
    return rows,anchors,stable,variables,parent


def run(root, *, output_override=None):
    path=root/CONFIG
    config=yaml.safe_load(path.read_text())
    require(config["authorization"]=={"preparation":True,"model_training":False,"model_prediction":False,
                "candidate_generation":False,"physics_evaluation":False,"confirmation_content_access":False},"Preparation authorization changed")
    require(config["condition"]["axes"]==OUTPUTS and config["condition"]["dimension"]==3 and
            config["training"]["seeds"]==[42,43,44] and config["validation_bank"]["draws_per_row"]==1,
            "Training interface/noise contract changed")
    output=output_override or within(root,config["artifacts"]["output_root"])
    require(not output.exists(),"Refusing to overwrite training preparation")
    rows,anchors,stable,variables,parent=verified_sources(root,config)
    train,val,excluded,transform,decoders=prepare_views(rows,anchors,stable,variables)
    require((len(train),len(val),len(excluded))==(12153,3110,1633),"Eligibility counts changed")
    training_protocol={"protocol":config,"parent_models":parent["models"],"external_condition_fields":OUTPUTS,
                       "training_authorized":False,"model_implementation_not_yet_locked":True}
    payloads={"Training_Protocol.json":canonical(training_protocol)+b"\n","Condition_Transform.json":canonical(transform)+b"\n",
              "Expert_Decoders.json":canonical(decoders)+b"\n"}
    for name,content in (("Train_Registry.jsonl",train),("Validation_Registry.jsonl",val),("Excluded_Registry.jsonl",excluded)):
        payloads[name]=b"".join(canonical(r)+b"\n" for r in content)
    for seed in config["training"]["seeds"]:
        payloads[f"Validation_Noise_seed{seed}.jsonl"]=b"".join(canonical(r)+b"\n" for r in validation_noise(val,seed))
    weights=np.array([r["root_equal_row_weight"] for r in train])/len(train)
    conditions=np.array([r["condition"] for r in train])
    mean=np.sum(weights[:,None]*conditions,axis=0)
    variance=np.sum(weights[:,None]*conditions**2,axis=0)
    require(np.allclose(mean,0,atol=1e-10) and np.allclose(variance,1,atol=1e-10),"Root-equal transform audit failed")
    lock={"status":"G1_G2_TRAINING_PREPARATION_PASS","protocol_sha256":file_hash(path),"implementation_sha256":file_hash(Path(__file__)),
          "cli_sha256":file_hash(root/"scripts/run_geometry19_performance_gated_training_preparation.py"),
          "environment":{"python":platform.python_version(),"numpy":np.__version__},"inputs":config["inputs"],
          "physics":config["physics"],"rows":{"train":len(train),"validation":len(val),"excluded":len(excluded)},
          "root_count":33,"condition_dim":3,"weighted_train_condition_mean":mean.tolist(),"weighted_train_condition_second_moment":variance.tolist(),
          "train_validation_pair_overlap":0,"decoder_round_trip_verified":True,"target_or_confirmation_content_parsed":False,
          "model_arrays_loaded":False,"physical_evaluations":0,"training_executed":False,"authorization":config["authorization"],
          "artifacts":{name: {"sha256":digest_bytes(data)} for name,data in payloads.items()}}
    report=("# G1/G2 training preparation\n\nStatus: `G1_G2_TRAINING_PREPARATION_PASS`.\n\n"
            "33 roots; 12,153 eligible training, 3,110 eligible validation, 1,633 excluded rows retained in original ledger.\n"
            "Original pair split preserved. Exactly three performance conditions; train-only root-equal log mean/std.\n"
            "Local 16D coordinates use frozen medoid/radius; pooled coordinates use absolute schema scaling.\n"
            "Three immutable validation noise/time banks, one for each seed, shared by local and pooled models.\n"
            "No target/confirmation content parsing, model loading, physics calls, training or generation.\n\n"
            "Next: implement and lock gate/CFM/pooled models, tests and training runtime, then separately authorize training.\n").encode()
    payloads["Preparation_Report.md"]=report
    lock["artifacts"]["Preparation_Report.md"]={"sha256":digest_bytes(report)}
    output.mkdir(parents=True,exist_ok=False)
    for name,data in payloads.items():write_once(output/name,data)
    write_json(output/"Training_Lock.json",lock)
    return lock


def digest_bytes(data):
    from hashlib import sha256
    return sha256(data).hexdigest()
