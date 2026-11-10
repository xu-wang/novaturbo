"""Isolated durable V3 UI jobs. Workers run the frozen public CLI on GPU 6."""
from __future__ import annotations
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from .geometry22_results import Geometry22Results, Geometry22ResultError


class Geometry22Jobs:
    def __init__(self, repository, root, previews):
        self.repository = Path(repository).resolve()
        self.root = Path(root).resolve()
        self.results = Geometry22Results(repository, self.root, previews)
        self.lock = threading.RLock()
        self.active = set()

    def directory(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{64}", job_id):
            raise FileNotFoundError("Unknown V3 job")
        path = (self.root / job_id).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("V3 job path escapes configured root")
        return path

    def save(self, job):
        directory = self.directory(job["job_id"])
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / (".job-" + uuid.uuid4().hex)
        with tmp.open("w") as stream:
            json.dump(job, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, directory / "job.json")

    def snapshot(self, job_id):
        with self.lock:
            path = self.directory(job_id)
            job = json.loads((path / "job.json").read_text())
            if job.get("job_id") != job_id or job.get("model") != "model V3":
                raise ValueError("V3 job identity changed")
            if job_id not in self.active and job["state"] in {"queued", "running"}:
                # Another server process may still own this job; never mark it interrupted.
                with (path / "worker.lock").open("a+") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return job
                    if (path / "run/Complete_Manifest.json").is_file():
                        try:
                            self.results.result(job_id)
                            job["state"] = "complete"
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            job.update(state="failed", error=str(exc))
                    else:
                        job.update(state="interrupted", error="Server/worker stopped; artifacts retained. No automatic rerun.")
                    self.save(job)
            if job["state"] == "complete":
                try:
                    self.results.result(job_id)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    job.update(state="failed", error=str(exc))
                    self.save(job)
            return job

    def list_jobs(self):
        if not self.root.exists():
            return []
        jobs = []
        for path in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if re.fullmatch(r"[0-9a-f]{64}", path.name) and (path / "job.json").is_file():
                jobs.append(self.snapshot(path.name))
        return jobs

    def submit(self, payload):
        names = {"thrust_N", "tsfc_kg_N_s", "mass_kg"}
        if not isinstance(payload, dict) or set(payload) != names:
            raise ValueError("Provide only thrust_N, tsfc_kg_N_s and mass_kg")
        if any(isinstance(v, bool) or v is None or isinstance(v, (list, dict)) for v in payload.values()):
            raise ValueError("Targets must be finite positive numbers")
        target = {name: float(payload[name]) for name in names}
        if not all(math.isfinite(v) and v > 0 for v in target.values()):
            raise ValueError("Targets must be finite positive numbers")
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            # A file lock serializes submission across Flask instances.
            with (self.root / ".submission.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if any(j["state"] in {"queued", "running"} for j in self.list_jobs()):
                    raise Geometry22ResultError("A V3 job is already running")
                job_id = uuid.uuid4().hex + uuid.uuid4().hex
                job = {"job_id": job_id, "state": "queued", "target": target,
                       "model": "model V3", "pipeline_profile": "v3_guided_cfm_development", "physical_gpu": 6}
                self.active.add(job_id)
                self.directory(job_id).mkdir(parents=True)
                worker_lock = (self.directory(job_id) / "worker.lock").open("a+")
                fcntl.flock(worker_lock, fcntl.LOCK_EX)
                try:
                    self.save(job)
                    threading.Thread(target=self.run, args=(job_id, worker_lock), daemon=True).start()
                except Exception:
                    worker_lock.close()
                    self.active.discard(job_id)
                    job.update(state="failed", error="Could not start worker")
                    self.save(job)
                    raise
                return job

    def run(self, job_id, worker_lock):
        path = self.directory(job_id)
        with worker_lock:
            job = json.loads((path / "job.json").read_text())
            try:
                job["state"] = "running"
                self.save(job)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="6", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                # Check physical identity and actual restricted PyTorch context; no CPU fallback.
                gpu = subprocess.run(["nvidia-smi", "-i", "6", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                                     env=env, capture_output=True, text=True, check=True)
                probe = subprocess.run([sys.executable, "-c", "import torch; available=torch.cuda.is_available() and torch.cuda.device_count()==1;\nif not available: raise RuntimeError('GPU 6 CUDA context unavailable')\nprint(torch.cuda.get_device_name(0))"],
                                       env=env, capture_output=True, text=True, check=True)
                job["gpu_identity"] = gpu.stdout.strip()
                job["torch_device"] = probe.stdout.strip()
                target = job["target"]
                command = [sys.executable, str(self.repository / "scripts/run_wp19_23_pipeline.py"),
                           "--model", "model V3", "--pipeline-profile", "v3_guided_cfm_development",
                           "--thrust-N", str(target["thrust_N"]), "--tsfc-kg-N-s", str(target["tsfc_kg_N_s"]),
                           "--mass-kg", str(target["mass_kg"]), "--device", "cuda:0", "--output-root", str(path / "run")]
                job["command"] = command
                self.save(job)
                with (path / "worker.log").open("w") as output:
                    process = subprocess.run(command, cwd=self.repository, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
                if process.returncode:
                    raise RuntimeError(f"V3 CLI failed ({process.returncode}): " + (path / "worker.log").read_text()[-1500:])
                data = self.results.result(job_id)
                if data["target_performance"] != target:
                    raise ValueError("V3 result target differs from submitted target")
                # NO_HIT/abstention are completed diagnoses, not worker errors.
                job.update(state="complete", delivery_status=data["delivery_status"], route_status=data["route_status"])
            except Exception as exc:
                job.update(state="failed", error=str(exc))
            finally:
                with self.lock:
                    self.save(job)
                    self.active.discard(job_id)

    def result(self, job_id):
        job = self.snapshot(job_id)
        if job["state"] != "complete":
            raise Geometry22ResultError("V3 job is not complete")
        payload = self.results.result(job_id)
        if payload["target_performance"] != job["target"]:
            raise Geometry22ResultError("V3 result target differs from submitted target")
        return payload | {"job_id": job_id}
