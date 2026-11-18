"""Public API of the standalone NovaTurbo V2.1 (Geometry19, 19-D) capability.

Four things can be done from this module:

* ``predict(design)``        forward surrogate: 19-D design -> thrust / TSFC / mass
* ``evaluate(design)``       V2.1 closed low-order physics: the authoritative answer
* ``generate(target, n)``    conditional CFM-4 generator: targets -> 19-D designs
* ``inverse_design(target)`` generate -> surrogate ranking -> V2.1 verification

Everything is local: no network, no repository checkout, no audit ledger. The
authoritative physics is the same code the NovaTurbo pipeline uses, and the
two networks are the frozen V2.1 checkpoints shipped in ``_vendor/data``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ._vendor.src.ai.generators.base import load_generator
from ._vendor.src.ai.geometry19_surrogate import load_weight_bundle
from ._vendor.src.physics.constraint_profiles import assess_constraint_profile
from ._vendor.src.physics.engine_match import Environment
from ._vendor.src.physics.geometry_state import Geometry19
from ._vendor.src.physics.runtime import build_physics_runtime

OUTPUT_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
PROTOCOL_RELATIVE = "config/physics/wp19_23_inverse_closure_v1.yaml"
DEFAULT_RESOURCE_ROOT = Path(__file__).resolve().parent / "_vendor"
GENERATOR_VARIANT = "bounded-sigmoid"
FORWARD_VARIANT = "performance3_log"


class V21Error(RuntimeError):
    """Raised when a request violates the V2.1 contract."""


def _largest_power_of_two_not_above(value: int) -> int:
    if value <= 0:
        return 0
    return 1 << (int(value).bit_length() - 1)


def _as_target(target: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(target, Mapping):
        raise V21Error("target must be a mapping with thrust_N, tsfc_kg_N_s, mass_kg")
    missing = [name for name in OUTPUT_NAMES if name not in target]
    if missing:
        raise V21Error(f"target is missing {missing}")
    values = {name: float(target[name]) for name in OUTPUT_NAMES}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise V21Error("target values must be finite and positive")
    return values


class V21Capability:
    """One loaded V2.1 capability: physics runtime + generator + surrogate."""

    def __init__(
        self,
        device: str = "cpu",
        resource_root: str | Path | None = None,
    ) -> None:
        self.resource_root = (
            Path(resource_root).resolve() if resource_root else DEFAULT_RESOURCE_ROOT
        )
        if not self.resource_root.is_dir():
            raise V21Error(f"resource root not found: {self.resource_root}")
        self.device = torch.device(device)
        protocol_path = self.resource_root / PROTOCOL_RELATIVE
        self.protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

        self.runtime = build_physics_runtime(self.resource_root, self.protocol)
        schema = self.runtime.design_schema
        variables = list(schema["design_variables"])
        self.feature_names: tuple[str, ...] = tuple(str(item["name"]) for item in variables)
        self.bounds = {
            str(item["name"]): (float(item["bounds"][0]), float(item["bounds"][1]))
            for item in variables
        }
        self.integer_names = frozenset(
            str(item["name"])
            for item in variables
            if str(item.get("dtype", "")).startswith("int")
        )
        self.reference_design = {
            str(item["name"]): (
                int(item["reference_value"])
                if str(item.get("dtype", "")).startswith("int")
                else float(item["reference_value"])
            )
            for item in variables
        }
        acceptance = self.protocol["diagnostic_acceptance"]
        self.tolerances = {
            name: float(acceptance["relative_tolerances"][name]) for name in OUTPUT_NAMES
        }
        self.epsilons = {
            name: float(acceptance["epsilons"][name]) for name in OUTPUT_NAMES
        }

        forward = self.protocol["models"]["forward"]["variants"][FORWARD_VARIANT]
        self.surrogate, self._normalization, self.surrogate_manifest = load_weight_bundle(
            self.resource_root / forward["weights"],
            expected_model_tag=forward["model_tag"],
            expected_feature_names=self.feature_names,
            expected_target_names=OUTPUT_NAMES,
            device=self.device,
        )
        self.output_transform = str(
            self.surrogate_manifest.get("output_transform", forward["output_transform"])
        )

        generator = self.protocol["models"]["generator"]["variants"][GENERATOR_VARIANT]
        self.generator = load_generator(
            self.resource_root / generator["checkpoint"], device=str(self.device)
        )
        normalizer = self.generator.normalizer
        if normalizer is None:
            raise V21Error("generator checkpoint carries no design normalizer")
        self._design_lower = torch.tensor(
            [float(item["physical_min"]) for item in normalizer.design],
            dtype=torch.float32,
            device=self.device,
        )
        self._design_upper = torch.tensor(
            [float(item["physical_max"]) for item in normalizer.design],
            dtype=torch.float32,
            device=self.device,
        )
        self._dflow = dict(self.protocol.get("dflow", {}))
        self._smooth_max_alpha = float(
            self.protocol.get("hit_alignment", {}).get("smooth_max_alpha", 12.0)
        )

    # ------------------------------------------------------------------ info
    def describe(self) -> dict[str, Any]:
        """Return the contracts a caller needs to build valid requests."""

        return {
            "capability": "novaturbo_v21_geometry19",
            "design_dimension": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "integer_features": sorted(self.integer_names),
            "bounds": {name: list(value) for name, value in self.bounds.items()},
            "reference_design": dict(self.reference_design),
            "outputs": list(OUTPUT_NAMES),
            "relative_tolerances": dict(self.tolerances),
            "physics": {
                "backend_id": self.runtime.backend_id,
                "physics_bundle_hash": self.runtime.physics_bundle_hash,
                "model_tag": self.protocol["physics_evaluation"]["model_tag"],
                "formal_use_allowed": bool(self.protocol.get("formal_use_allowed", False)),
            },
            "generator": {
                "variant": GENERATOR_VARIANT,
                "model_tag": generator_tag(self.protocol),
                "conditioning": list(OUTPUT_NAMES),
            },
            "surrogate": {
                "variant": FORWARD_VARIANT,
                "model_tag": self.surrogate_manifest.get("model_tag"),
                "output_transform": self.output_transform,
            },
        }

    # -------------------------------------------------------------- design I/O
    def parse_design(self, design: Mapping[str, Any]) -> Geometry19:
        if not isinstance(design, Mapping):
            raise V21Error("design must be a mapping of the 19 Geometry19 coordinates")
        missing = [name for name in self.feature_names if name not in design]
        if missing:
            raise V21Error(f"design is missing coordinates: {missing}")
        unknown = [name for name in design if name not in self.feature_names]
        if unknown:
            raise V21Error(f"design has unknown coordinates: {unknown}")
        values = {name: design[name] for name in self.feature_names}
        for name in self.integer_names:
            values[name] = int(round(float(values[name])))
        try:
            return Geometry19.from_mapping(values)
        except Exception as exc:  # noqa: BLE001 - surface the model's own message
            raise V21Error(f"design rejected by the Geometry19 schema: {exc}") from exc

    def _design_row(self, design: Mapping[str, Any]) -> np.ndarray:
        parsed = self.parse_design(design)
        mapping = parsed.as_mapping()
        return np.asarray([[float(mapping[name]) for name in self.feature_names]], dtype=np.float32)

    # -------------------------------------------------------------- surrogate
    def predict(
        self,
        design: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        ) -> dict[str, float] | list[dict[str, float]]:
        """Forward surrogate prediction (fast, not authoritative)."""

        single = isinstance(design, Mapping)
        mappings = [design] if single else list(design)
        rows = np.concatenate([self._design_row(item) for item in mappings], axis=0)
        normalized = (rows - self._normalization["input_mean"]) / self._normalization["input_std"]
        with torch.no_grad():
            raw = self.surrogate(
                torch.from_numpy(normalized.astype(np.float32)).to(self.device)
            ).cpu().numpy()
        values = raw * self._normalization["output_std"] + self._normalization["output_mean"]
        if self.output_transform == "natural_log":
            values = np.exp(values)
        elif self.output_transform != "identity":
            raise V21Error(f"unsupported surrogate output transform: {self.output_transform}")
        results = [
            {name: float(values[row, index]) for index, name in enumerate(OUTPUT_NAMES)}
            for row in range(values.shape[0])
        ]
        return results[0] if single else results

    # --------------------------------------------------------------- physics
    def evaluate(
        self,
        design: Mapping[str, Any],
        environment: Mapping[str, Any] | None = None,
        target: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Authoritative V2.1 physics evaluation of one 19-D design."""

        parsed = self.parse_design(design)
        env = Environment(**dict(environment)) if environment else Environment()
        result = self.runtime.evaluate(parsed, env)
        assessment = assess_constraint_profile(
            result,
            self.runtime.constraint_profile,
            conservation_tolerances=self.runtime.conservation_tolerances,
        )
        performance = (
            {name: float(result.performance[name]) for name in OUTPUT_NAMES}
            if result.performance
            else None
        )
        payload: dict[str, Any] = {
            "design": parsed.as_mapping(),
            "performance": performance,
            "solver": {
                "converged": bool(result.solver.converged),
                "failure_code": (
                    result.solver.failure_code.value
                    if result.solver.failure_code is not None
                    else None
                ),
                "message": result.solver.message,
                "flow_root_count": int(result.operating_point.get("flow_root_count", 0)),
            },
            "constraints": assessment.as_mapping(),
            "physics": {
                "backend_id": self.runtime.backend_id,
                "physics_bundle_hash": self.runtime.physics_bundle_hash,
                "calibration_pack_hash": self.runtime.model_config.calibration_pack_hash,
            },
        }
        if target is not None:
            target_values = _as_target(target)
            if performance is None:
                payload["hit"] = {
                    "rho": None,
                    "failed_axes": None,
                    "strict_hit": False,
                    "reason": "NO_PERFORMANCE",
                }
            else:
                payload["hit"] = self.hit_metrics(performance, target_values)
        return payload

    # ------------------------------------------------------------- hit metric
    def hit_metrics(
        self, performance: Mapping[str, Any], target: Mapping[str, Any]
    ) -> dict[str, Any]:
        target_values = _as_target(target)
        ratios = {}
        for name in OUTPUT_NAMES:
            denominator = max(abs(target_values[name]), self.epsilons[name])
            ratios[name] = abs(float(performance[name]) - target_values[name]) / denominator
        scaled = {name: ratios[name] / self.tolerances[name] for name in OUTPUT_NAMES}
        failed = sum(value > 1.0 for value in scaled.values())
        return {
            "axis_ratios": scaled,
            "failed_axes": failed,
            "rho": max(scaled.values()),
            "sum_axis_ratios": sum(scaled.values()),
            "strict_hit": failed == 0,
        }

    # ------------------------------------------------------------- generator
    def generate(
        self,
        target: Mapping[str, Any],
        n_samples: int = 256,
        seed: int | None = None,
        guided: bool = False,
    ) -> list[dict[str, float]]:
        """Sample 19-D designs conditioned on the three performance targets.

        ``guided=False`` is plain CFM-4 sampling. ``guided=True`` runs the frozen
        model's D-Flow source guidance: the latent sources are moved with the
        surrogate's hit-aligned objective for a few Adam steps before integration,
        which is how the NovaTurbo pipeline makes candidates target-directed.
        """

        values = _as_target(target)
        if guided:
            return self._guided_generate(values, n_samples, seed)
        condition = [[values[name] for name in OUTPUT_NAMES]]
        samples = self.generator.sample(
            condition, int(n_samples), seed=seed, space="physical"
        )
        array = samples.values.detach().cpu().numpy().reshape(-1, len(self.feature_names))
        return self._designs_from_array(array)

    def _designs_from_array(self, array: np.ndarray) -> list[dict[str, float]]:
        designs = []
        for row in array:
            design = {name: float(row[index]) for index, name in enumerate(self.feature_names)}
            for name in self.integer_names:
                design[name] = int(round(design[name]))
            designs.append(design)
        return designs

    def _normalize_design_tensor(self, physical: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self._normalization["input_mean"], dtype=physical.dtype, device=physical.device
        )
        std = torch.as_tensor(
            self._normalization["input_std"], dtype=physical.dtype, device=physical.device
        )
        return (physical - mean) / std

    def _integrate_sources(
        self, sources: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Integrate explicit latent sources through the frozen CFM-4 RK4 field."""

        from torchdiffeq import odeint

        generator = self.generator
        steps = int(generator.model_config.get("ode_steps", 32))
        if steps <= 0:
            raise V21Error("CFM checkpoint declares an invalid ODE step count")
        expanded = condition if condition.shape[0] == sources.shape[0] else condition.expand(
            sources.shape[0], -1
        )

        def velocity(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return generator.velocity(time, state, expanded)

        trajectory = odeint(
            velocity,
            sources,
            torch.tensor([0.0, 1.0], dtype=sources.dtype, device=sources.device),
            method="rk4",
            options={"step_size": 1.0 / steps},
            rtol=float(generator.model_config.get("ode_rtol", 1.0e-5)),
            atol=float(generator.model_config.get("ode_atol", 1.0e-5)),
        )
        normalized, _, _ = generator._from_model_space(trajectory[-1])
        return normalized

    def _guided_generate(
        self,
        target_values: Mapping[str, float],
        n_samples: int,
        seed: int | None,
    ) -> list[dict[str, float]]:
        """D-Flow source guidance inside the frozen CFM-4 (V2.1 protocol config)."""

        generator = self.generator
        normalizer = generator.normalizer
        device = generator.device
        steps = int(self._dflow.get("steps", 24))
        learning_rate = float(self._dflow.get("learning_rate", 0.01))
        prior_weight = float(self._dflow.get("source_prior_weight", 0.01))
        bound_penalty = float(self._dflow.get("design_bound_penalty", 5.0))
        max_displacement = float(self._dflow.get("maximum_source_displacement_linf", 0.75))
        clip_norm = float(self._dflow.get("gradient_clip_norm", 1.0))

        condition = torch.as_tensor(
            normalizer.transform_conditions([[target_values[name] for name in OUTPUT_NAMES]]),
            dtype=torch.float32,
            device=device,
        )
        random_generator = torch.Generator(device=device)
        if seed is not None:
            random_generator.manual_seed(int(seed))
        initial = torch.randn(
            int(n_samples),
            len(self.feature_names),
            generator=random_generator,
            device=device,
            dtype=torch.float32,
        )
        sources = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([sources], lr=learning_rate)

        target_tensor = torch.tensor(
            [target_values[name] for name in OUTPUT_NAMES], dtype=torch.float32, device=device
        )
        tolerance_tensor = torch.tensor(
            [self.tolerances[name] for name in OUTPUT_NAMES],
            dtype=torch.float32,
            device=device,
        )
        epsilon_tensor = torch.tensor(
            [self.epsilons[name] for name in OUTPUT_NAMES],
            dtype=torch.float32,
            device=device,
        )
        output_mean = torch.as_tensor(
            self._normalization["output_mean"], dtype=torch.float32, device=device
        )
        output_std = torch.as_tensor(
            self._normalization["output_std"], dtype=torch.float32, device=device
        )

        generator.eval()
        self.surrogate.eval()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            normalized = self._integrate_sources(sources, condition)
            physical = self._design_lower + 0.5 * (normalized + 1.0) * (
                self._design_upper - self._design_lower
            )
            raw = self.surrogate(self._normalize_design_tensor(physical))
            prediction = raw * output_std + output_mean
            if self.output_transform == "natural_log":
                prediction = torch.exp(prediction)
            ratio = (prediction[:, :3] - target_tensor).abs() / (
                tolerance_tensor * torch.maximum(target_tensor.abs(), epsilon_tensor)
            )
            smooth_rho = torch.logsumexp(self._smooth_max_alpha * ratio, dim=1) / (
                self._smooth_max_alpha
            )
            prior = (sources - initial).square().mean(dim=1)
            boundary = torch.relu(normalized.abs() - 1.0).square().mean(dim=1)
            loss = (smooth_rho + prior_weight * prior + bound_penalty * boundary).mean()
            if not torch.isfinite(loss):
                raise V21Error("D-Flow objective became non-finite")
            loss.backward()
            if sources.grad is None or not torch.isfinite(sources.grad).all():
                raise V21Error("D-Flow gradient became non-finite")
            torch.nn.utils.clip_grad_norm_([sources], clip_norm)
            optimizer.step()
            with torch.no_grad():
                displacement = (sources - initial).clamp(-max_displacement, max_displacement)
                sources.copy_(initial + displacement)

        with torch.no_grad():
            normalized = self._integrate_sources(sources.detach(), condition)
            physical = self._design_lower + 0.5 * (normalized + 1.0) * (
                self._design_upper - self._design_lower
            )
        return self._designs_from_array(physical.detach().cpu().numpy())

    # -------------------------------------------------------- inverse design
    def inverse_design(
        self,
        target: Mapping[str, Any],
        n_samples: int = 256,
        seed: int | None = None,
        physics_budget: int | None = None,
        guided: bool = True,
    ) -> dict[str, Any]:
        """Generate, rank with the surrogate, then verify candidates with V2.1.

        This is the V2.1 *capability* without the pipeline's ledger, D-stage
        surrogate-gradient refinement and NSGA-II rescue stages. By default every
        retained candidate receives a true V2.1 evaluation (``N_keep`` is a power
        of two, usually 256); the scan stops early as soon as a physics-accepted
        strict hit is found. ``physics_budget`` caps the number of V2.1 calls.
        """

        target_values = _as_target(target)
        designs = self.generate(
            target_values, n_samples=n_samples, seed=seed, guided=guided
        )
        in_bounds = []
        for design in designs:
            if all(
                self.bounds[name][0] <= design[name] <= self.bounds[name][1]
                for name in self.feature_names
            ):
                in_bounds.append(design)
        kept = _largest_power_of_two_not_above(len(in_bounds))
        if kept == 0:
            return {
                "target": target_values,
                "status": "NO_VALID_CANDIDATE",
                "delivered": None,
                "generated": len(designs),
                "in_bounds": 0,
            }
        candidates = in_bounds[:kept]
        predictions = self.predict(candidates)
        ranked = sorted(
            zip(candidates, predictions),
            key=lambda pair: self.hit_metrics(pair[1], target_values)["rho"],
        )
        budget = (
            len(ranked) if not physics_budget else min(max(1, int(physics_budget)), len(ranked))
        )
        evaluated = []
        for design, prediction in ranked[:budget]:
            result = self.evaluate(design, target=target_values)
            row = {
                "design": design,
                "predicted": prediction,
                "predicted_rho": self.hit_metrics(prediction, target_values)["rho"],
                "performance": result["performance"],
                "hit": result["hit"],
                "solver_converged": result["solver"]["converged"],
                "physics_accepted": bool(result["constraints"]["accepted"]),
            }
            evaluated.append(row)
            hit = row["hit"]
            if (
                hit["rho"] is not None
                and hit["strict_hit"]
                and row["physics_accepted"]
            ):
                break
        acceptable = [
            row
            for row in evaluated
            if row["physics_accepted"] and row["hit"]["rho"] is not None
        ]
        pool = acceptable or [
            row for row in evaluated if row["hit"]["rho"] is not None
        ]
        if not pool:
            return {
                "target": target_values,
                "status": "NO_PHYSICS_VALID_CANDIDATE",
                "delivered": None,
                "generated": len(designs),
                "in_bounds": len(in_bounds),
                "kept": kept,
                "evaluated": evaluated,
                "physics_calls": len(evaluated),
            }
        best = min(
            pool,
            key=lambda row: (
                row["hit"]["failed_axes"],
                row["hit"]["rho"],
                row["hit"]["sum_axis_ratios"],
            ),
        )
        return {
            "target": target_values,
            "status": (
                "DIAGNOSTIC_HIT"
                if best["hit"]["strict_hit"] and best["physics_accepted"]
                else "BEST_APPROXIMATION_NOT_HIT"
            ),
            "delivered": best,
            "generated": len(designs),
            "in_bounds": len(in_bounds),
            "kept": kept,
            "physics_calls": len(evaluated),
            "ranking": [
                {
                    "predicted_rho": row["predicted_rho"],
                    "true_rho": row["hit"]["rho"],
                    "strict_hit": row["hit"]["strict_hit"],
                    "physics_accepted": row["physics_accepted"],
                }
                for row in evaluated
            ],
        }

    # ------------------------------------------------------------------ mesh
    def export_mesh(
        self,
        design: Mapping[str, Any],
        output_directory: str | Path,
        quality: str = "preview_mesh",
    ) -> dict[str, Any]:
        """Export the component STL set (requires the bundle built with --mesh)."""

        try:
            from ._vendor.src.visualization.geometry19_adapter import (
                geometry19_to_assembly_params,
            )
            from ._vendor.src.visualization.geometry19_mesh import (
                generate_geometry19_meshes,
            )
        except ImportError as exc:
            raise V21Error(
                "this bundle was built without --mesh; rebuild to enable STL export"
            ) from exc
        parsed = self.parse_design(design)
        assembly = geometry19_to_assembly_params(parsed, self.runtime.model_config)
        result = generate_geometry19_meshes(assembly, output_directory, quality=quality)
        return result.as_mapping()


def generator_tag(protocol: Mapping[str, Any]) -> str | None:
    variants = protocol.get("models", {}).get("generator", {}).get("variants", {})
    declared = variants.get(GENERATOR_VARIANT, {})
    return declared.get("model_tag") if isinstance(declared, Mapping) else None


def load_capability(device: str = "cpu", resource_root: str | Path | None = None) -> V21Capability:
    """Convenience constructor mirroring ``V21Capability(...)``."""

    return V21Capability(device=device, resource_root=resource_root)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
