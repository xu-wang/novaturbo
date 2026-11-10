"""Locked G1/G2 training runner. No generation, surrogate loading, or physics."""
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import torch
from torch.nn import functional as F
import yaml

from src.ai.geometry19_performance_gated_protocol import file_hash, read_rows, require, within
from src.ai.geometry19_performance_gated_physics import write_json, write_once
from src.ai.geometry19_performance_gated_models import PerformanceGate, LocalCFM, PooledCFM

CONFIG = "config/experiments/geometry19_performance_gated_implementation_v1.yaml"
SOURCES = [CONFIG, "src/ai/geometry19_performance_gated_models.py",
           "src/ai/geometry19_performance_gated_training.py",
           "src/ai/geometry19_performance_gated_protocol.py",
           "src/ai/geometry19_performance_gated_physics.py",
           "scripts/run_geometry19_performance_gated_training.py",
           "tests/test_geometry19_performance_gated_models.py"]


def verified_data(root):
    config = yaml.safe_load((root/CONFIG).read_text())
    spec = config["preparation_lock"]
    lock_path = within(root, spec["path"])
    require(file_hash(lock_path) == spec["sha256"], "Preparation lock changed")
    lock = json.loads(lock_path.read_text())
    require(lock["status"] == "G1_G2_TRAINING_PREPARATION_PASS", "Preparation not passed")
    base = lock_path.parent
    for name, item in lock["artifacts"].items():
        require(file_hash(within(base, name)) == item["sha256"], f"Preparation artifact changed: {name}")
    for name, item in lock["inputs"].items():
        require(file_hash(within(root, item["path"])) == item["sha256"], f"Label source changed: {name}")
    for name, key in [("config/experiments/geometry19_performance_gated_training_v1.yaml", "protocol_sha256"),
                      ("src/ai/geometry19_performance_gated_training_preparation.py", "implementation_sha256"),
                      ("scripts/run_geometry19_performance_gated_training_preparation.py", "cli_sha256")]:
        require(file_hash(root/name) == lock[key], f"Preparation code changed: {name}")
    protocol = json.loads((base/"Training_Protocol.json").read_text())
    decoders = json.loads((base/"Expert_Decoders.json").read_text())
    train, val = read_rows(base/"Train_Registry.jsonl"), read_rows(base/"Validation_Registry.jsonl")
    for rows, split, count in [(train, "train", 12153), (val, "validation", 3110)]:
        require(len(rows) == count and all(r["split"] == split and len(r["condition"]) == 3 for r in rows), "Data interface changed")
        require(set(r["supervision"]["root_index"] for r in rows) == set(range(33)), "Missing root")
    require({r["pair_id"] for r in train}.isdisjoint(r["pair_id"] for r in val), "Pair leakage")
    return config, base, protocol, decoders, train, val


def category_counts(decoders):
    return [len(decoders["integer_categories"][axis]) for axis in decoders["integer_axes"]]


def build_model(arm, counts):
    if arm == "gate": return PerformanceGate()
    if arm == "local_cfm": return LocalCFM()
    if arm == "pooled_cfm": return PooledCFM(counts)
    raise ValueError(f"Unknown arm: {arm}")


def seal(root, test_report):
    config, _, _, decoders, _, _ = verified_data(root)
    report = ET.parse(test_report).getroot()
    suites = [report] if report.tag == "testsuite" else list(report.iter("testsuite"))
    require(bool(suites) and sum(int(s.get("tests", 0)) for s in suites) >= 10 and
            all(int(s.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped")), "Tests not passed")
    payload = {"status": "G1_G2_IMPLEMENTATION_LOCKED_NOT_TRAINED", "preparation_lock": config["preparation_lock"],
               "sources": {name: file_hash(root/name) for name in SOURCES},
               "test_report": {"path": str(test_report.resolve().relative_to(root)), "sha256": file_hash(test_report)},
               "parameter_counts": {arm: sum(p.numel() for p in build_model(arm, category_counts(decoders)).parameters())
                                    for arm in config["runtime"]["arms"]},
               "formal_training_executed": False, "candidate_generation": False, "physics_evaluations": 0}
    path = within(root, config["artifacts"]["implementation_lock"])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return payload


def verify_implementation(root, config):
    path = within(root, config["artifacts"]["implementation_lock"])
    lock = json.loads(path.read_text())
    require(lock["status"] == "G1_G2_IMPLEMENTATION_LOCKED_NOT_TRAINED" and
            lock["preparation_lock"] == config["preparation_lock"], "Wrong implementation lock")
    require(set(lock["sources"]) == set(SOURCES), "Incomplete source lock")
    for name, expected in lock["sources"].items():
        require(file_hash(root/name) == expected, f"Implementation changed: {name}")
    report = lock["test_report"]
    require(file_hash(within(root, report["path"])) == report["sha256"], "Test evidence changed")
    return file_hash(path)


def cuda_manifest():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "6", "Set CUDA_VISIBLE_DEVICES=6; CPU fallback forbidden")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Physical GPU 6 is not visible")
    physical = subprocess.check_output(["nvidia-smi", "-i", "6", "--query-gpu=index,uuid,name,driver_version",
                                        "--format=csv,noheader"], text=True).strip()
    require(physical.split(",")[0].strip() == "6", "Wrong physical device")
    prop = torch.cuda.get_device_properties(0)
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "visible_devices": "6", "logical_device": "cuda:0",
            "physical_device": physical, "name": prop.name, "memory_bytes": prop.total_memory}


def set_determinism(seed):
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Set deterministic CUBLAS workspace before launch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensors(rows, device):
    def tensor(values, dtype=torch.float32): return torch.tensor(values, dtype=dtype, device=device)
    return {"condition": tensor([r["condition"] for r in rows]),
            "root": tensor([r["supervision"]["root_index"] for r in rows], torch.long),
            "local_cfm": tensor([r["local_state"] for r in rows]),
            "pooled_cfm": tensor([r["pooled_state"] for r in rows]),
            "integers": tensor([r["integer_class_indices"] for r in rows], torch.long),
            "weight": tensor([r["root_equal_row_weight"] for r in rows])}


def root_equal_mean(loss, roots):
    require(loss.ndim == 1 and roots.shape == loss.shape, "Bad per-row loss")
    return torch.stack([loss[roots == index].mean() for index in range(33)]).mean()


def flow_pair(target, epsilon, time):
    return (1-time[:, None])*epsilon + time[:, None]*target, target-epsilon


def row_losses(model, arm, batch, epsilon=None, time=None):
    c, roots = batch["condition"], batch["root"]
    if arm == "gate": return F.cross_entropy(model(c), roots, reduction="none")
    state, velocity = flow_pair(batch[arm], epsilon, time)
    predicted = model(state, time, c, roots) if arm == "local_cfm" else model(state, time, c)
    loss = (predicted-velocity).square().mean(dim=1)
    if arm == "pooled_cfm":
        ce = torch.stack([F.cross_entropy(logits, batch["integers"][:, k], reduction="none")
                          for k, logits in enumerate(model.integer_logits(c))], dim=1).mean(dim=1)
        loss = loss+ce
    return loss


def batches(roots, arm, generator, settings):
    if arm == "gate":
        order = torch.randperm(len(roots), generator=generator)
        yield from order.split(settings["batch_size"])
    else:
        groups = [torch.nonzero(roots.cpu() == k, as_tuple=True)[0] for k in range(33)]
        require(all(len(g) for g in groups), "Balanced sampling requires every root")
        for _ in range(settings["steps_per_epoch"]):
            yield torch.cat([g[torch.randint(len(g), (settings["rows_per_root_per_batch"],), generator=generator)] for g in groups])


def take(data, indices):
    return {key: value[indices] for key, value in data.items()}


@dataclass
class Selection:
    best: float = math.inf
    best_epoch: int = 0
    anchor: float = math.inf
    stale: int = 0

    def update(self, epoch, value, settings):
        require(math.isfinite(value), "Nonfinite validation objective")
        improved = value < self.best
        if improved: self.best, self.best_epoch = value, epoch
        if value < self.anchor-settings["min_delta"]:
            self.anchor, self.stale = value, 0
        else:
            self.stale += 1
        return improved, epoch >= settings["minimum_epochs"] and self.stale >= settings["patience"]


@torch.no_grad()
def validate(model, arm, data, bank):
    model.eval()
    losses = []
    for indices in torch.arange(len(data["root"]), device=data["root"].device).split(528):
        batch = take(data, indices)
        losses.append(row_losses(model, arm, batch, bank["epsilon"][indices], bank["time"][indices]))
    return float(root_equal_mean(torch.cat(losses), data["root"]))


@torch.no_grad()
def gate_diagnostics(model, data):
    model.eval()
    probabilities = model(data["condition"]).softmax(dim=1)
    top = probabilities.topk(2, dim=1).indices
    top1 = (top[:, 0] == data["root"]).float()
    top2 = (top == data["root"][:, None]).any(dim=1).float()
    confidence = probabilities.max(dim=1).values
    weight = data["weight"] / len(top1)
    ece = confidence.new_zeros(())
    for k in range(10):
        mask = (confidence >= k/10) & ((confidence < (k+1)/10) if k < 9 else (confidence <= 1))
        ece += (weight[mask]*(confidence[mask]-top1[mask])).sum().abs()
    per_root = [float(top2[data["root"] == k].mean()) for k in range(33)]
    macro1, macro2 = float(root_equal_mean(top1, data["root"])), float(root_equal_mean(top2, data["root"]))
    return {"macro_top1": macro1, "macro_top2": macro2, "per_root_top2": per_root,
            "root_equal_ece_10_bins": float(ece), "diagnostic_pass": macro1 >= .8 and macro2 >= .95 and min(per_root) >= .8,
            "diagnostic_only_not_training_block": True}


def fit(model, arm, train, val, bank, settings, seed, out, identity):
    """Internal loop; synthetic tests may supply tiny budgets, production uses frozen settings."""
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"Identity.json", identity)
    device = train["condition"].device
    sampler = torch.Generator().manual_seed(seed+10000)
    noise = torch.Generator(device=device).manual_seed(seed+20000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    selection = Selection()
    for epoch in range(1, settings["maximum_epochs"]+1):
        model.train()
        total, count, steps = 0., 0, 0
        for indices in batches(train["root"], arm, sampler, settings):
            batch = take(train, indices.to(device))
            n = len(indices)
            epsilon = torch.randn((n, 16), device=device, generator=noise) if arm != "gate" else None
            time = torch.rand(n, device=device, generator=noise) if arm != "gate" else None
            per_row = row_losses(model, arm, batch, epsilon, time)
            loss = (per_row*batch["weight"]).mean() if arm == "gate" else per_row.mean()
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require(bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()), "Nonfinite gradient")
            if arm != "gate": torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*n
            count += n
            steps += 1
        objective = validate(model, arm, val, bank)
        improved, stop = selection.update(epoch, objective, settings)
        record = {"epoch": epoch, "training_loss": total/count, "validation_objective": objective,
                  "steps": steps, "best_epoch": selection.best_epoch, "stale_epochs": selection.stale}
        write_json(out/f"epoch_{epoch:04d}.json", record)
        if improved:
            stream = io.BytesIO()
            torch.save({"identity": identity, "epoch": epoch, "validation_objective": objective,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, stream)
            write_once(out/f"best_epoch_{epoch:04d}.pt", stream.getvalue())
        print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
        if stop: break
    checkpoint = out/f"best_epoch_{selection.best_epoch:04d}.pt"
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    require(saved["identity"] == identity, "Checkpoint identity changed")
    model.load_state_dict(saved["state_dict"], strict=True)
    reloaded = validate(model, arm, val, bank)
    require(abs(reloaded-selection.best) <= 1e-7*max(1., abs(selection.best)), "Reload metric changed")
    summary = {"arm": arm, "seed": seed, "epochs": epoch, "best_epoch": selection.best_epoch,
               "best_validation_objective": selection.best, "reload_validation_objective": reloaded,
               "checkpoint": checkpoint.name, "checkpoint_sha256": file_hash(checkpoint)}
    if arm == "gate": summary["gate_diagnostics"] = gate_diagnostics(model, val)
    write_json(out/"Summary.json", summary)
    return summary


def execute(root, *, authorized=False, preflight=False):
    require(preflight or authorized, "Formal training requires explicit authorization")
    config, base, protocol, decoders, rows, validation = verified_data(root)
    implementation_hash = verify_implementation(root, config)
    environment = cuda_manifest()
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Deterministic CUBLAS setting missing")
    if preflight:
        return {"status": "TRAINING_PREFLIGHT_PASS", "environment": environment, "implementation_lock_sha256": implementation_hash,
                "training_executed": False}
    out = within(root, config["artifacts"]["training_output"])
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"implementation_lock_sha256": implementation_hash, "preparation_lock": config["preparation_lock"],
                "environment": environment, "authorization": "explicit_cli_training_only", "generation": False, "physics": False}
    write_json(out/"Runtime_Manifest.json", manifest)
    results = []
    try:
        train, val = tensors(rows, "cuda:0"), tensors(validation, "cuda:0")
        for seed in config["runtime"]["seeds"]:
            records = read_rows(base/f"Validation_Noise_seed{seed}.jsonl")
            require([r["design_hash"] for r in records] == [r["design_hash"] for r in validation], "Noise bank order mismatch")
            bank = {k: torch.tensor([r[k] for r in records], dtype=torch.float32, device="cuda:0") for k in ("epsilon", "time")}
            for arm in config["runtime"]["arms"]:
                set_determinism(seed)
                model = build_model(arm, category_counts(decoders)).to("cuda:0")
                settings = protocol["parent_models"]["gate" if arm == "gate" else "local_cfm"]
                identity = {"arm": arm, "seed": seed, "implementation_lock_sha256": implementation_hash,
                            "preparation_lock_sha256": config["preparation_lock"]["sha256"], "settings": settings}
                results.append(fit(model, arm, train, val, bank, settings, seed, out/f"seed{seed}"/arm, identity))
                del model
        summary = {"status": "G1_G2_TRAINING_COMPLETE", "runs": results, "run_count": len(results),
                   "generation_executed": False, "physics_evaluations": 0}
        write_json(out/"Training_Summary.json", summary)
        return summary
    except Exception as exc:
        write_json(out/"Failure.json", {"status": "TRAINING_FAILED_NO_AUTOMATIC_RESUME", "error": str(exc), "completed_runs": results})
        raise
