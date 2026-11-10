"""Opt-in FAR precision backend; original modules and profile registry stay immutable."""
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import FunctionType, MappingProxyType
import platform

import numpy as np
import scipy

from src.ai.geometry19_performance_gated_protocol import file_hash, require
from src.ai.geometry19_far_precision_diagnostic import digest, COMBUSTOR_HASH
from src.physics import combustor_map_enhanced as original

VERSION = "geometry19_v2_2_far_precision_v1"


def isolated_combustor(counters, *, precision=True):
    source = original.solve_combustor_map_enhanced
    solver = original.brentq
    namespace = dict(source.__globals__)
    def precision_root(function, lower, upper, **kwargs):
        require(kwargs == {"xtol": 1e-12, "rtol": 1e-12, "maxiter": 64, "full_output": True, "disp": False},
                "Unexpected FAR contract")
        counters["far_solver_invocations"] += 1
        def counted(x):
            counters["far_function_evaluations"] += 1
            return function(x)
        tolerance = 1e-15 if precision else 1e-12
        return solver(counted, lower, upper, **{**kwargs, "xtol": tolerance, "rtol": tolerance})
    namespace["brentq"] = precision_root
    cloned = FunctionType(source.__code__, namespace, source.__name__, source.__defaults__, source.__closure__)
    cloned.__kwdefaults__ = dict(source.__kwdefaults__ or {})
    return cloned


def tracked_parent(parent, *, precision=False):
    runtime = parent.physics_runtime
    counters = {"far_solver_invocations": 0, "far_function_evaluations": 0}
    cloned = isolated_combustor(counters, precision=precision)
    parent_solver = runtime.backend.solver
    def injected_solver(*args, **kwargs):
        supplied = kwargs.get("combustor_solver")
        require(isinstance(supplied, partial) and supplied.func is original.solve_combustor_map_enhanced and
                not supplied.args and set(supplied.keywords) == {"component_map", "thermo", "map_policy"}, "Combustor injection mismatch")
        return parent_solver(*args, **{**kwargs, "combustor_solver": partial(cloned, **supplied.keywords)})
    return replace(parent, physics_runtime=replace(runtime, backend=replace(runtime.backend, solver=injected_solver))), counters


def build_candidate(parent, root):
    require(file_hash(root / "src/physics/combustor_map_enhanced.py") == COMBUSTOR_HASH, "Original combustor changed")
    runtime = parent.physics_runtime
    require(runtime.model_version == "geometry19_map_enhanced_v2_2_full_variable_thermo_shadow", "Wrong parent backend")
    tracked, counters = tracked_parent(parent, precision=True)
    implementation = {"version": VERSION, "source_sha256": file_hash(Path(__file__)),
                      "original_combustor_sha256": COMBUSTOR_HASH,
                      "parent_context": parent.context_hash, "parent_bundle": runtime.physics_bundle_hash,
                      "parameters": {"xtol": 1e-15, "rtol": 1e-15, "maxiter": 64},
                      "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    bundle = digest(implementation)
    backend = replace(tracked.physics_runtime.backend, backend_id=VERSION + "_backend", model_version=VERSION)
    provenance = {**runtime.provenance_hashes, "far_precision_implementation": implementation["source_sha256"],
                  "parent_physics_bundle": runtime.physics_bundle_hash}
    candidate_runtime = replace(runtime, backend=backend, physics_bundle_hash=bundle,
                                provenance_hashes=MappingProxyType(provenance),
                                source_paths=MappingProxyType({**runtime.source_paths, "far_precision_implementation": Path(__file__).resolve()}))
    context = replace(parent, physics_runtime=candidate_runtime,
                      context_hash=digest({"parent_context": parent.context_hash, "candidate_bundle": bundle, "version": VERSION}),
                      provenance_hashes=MappingProxyType({**parent.provenance_hashes, "far_precision_bundle": bundle}))
    return context, counters, implementation
