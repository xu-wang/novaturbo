"""One resolved contract and one physics runtime shared by WP19-23 B--F."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from src.ai.geometry19_inverse_contract import WP1923Contract, load_wp19_23_contract
from src.ai.geometry19_pipeline_profiles import (
    PipelineProfile,
    PipelineProfileRegistry,
    load_pipeline_profile_registry,
)
from src.ai.geometry19_selection import ConstraintDominanceContract
from src.ai.geometry19_target_domain import (
    Geometry19TargetDomain,
    load_geometry19_target_domain,
)
from src.physics.runtime import PhysicsRuntime, PhysicsSolver, build_physics_runtime


class Geometry19PipelineContextError(ValueError):
    """Raised when a stage attempts to use a mismatched pipeline context."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PipelineContext:
    """Repository-scoped immutable dependencies shared by every pipeline stage."""

    repository_root: Path
    profile: PipelineProfile
    profile_registry_version: str
    contract_path: Path
    contract: WP1923Contract
    physics_runtime: PhysicsRuntime
    constraint_dominance: ConstraintDominanceContract
    target_domain: Geometry19TargetDomain | None
    provenance_hashes: Mapping[str, str]
    context_hash: str

    @property
    def protocol(self) -> Mapping[str, Any]:
        return self.contract.payload

    def contract_mapping(self) -> dict[str, Any]:
        return {
            "pipeline_profile": self.profile.contract_mapping(),
            "profile_registry_version": self.profile_registry_version,
            "protocol_path": str(self.contract_path),
            "context_hash": self.context_hash,
            "provenance_hashes": dict(self.provenance_hashes),
            "physics_runtime": self.physics_runtime.contract_mapping(),
            "target_domain": (
                None
                if self.target_domain is None
                else {
                    "domain_version": self.target_domain.domain_version,
                    "path": str(self.target_domain.source_path),
                    "sha256": self.target_domain.source_hash,
                }
            ),
        }


def build_pipeline_context(
    repository_root: str | Path,
    *,
    pipeline_profile: str | None = None,
    profile_registry: PipelineProfileRegistry | None = None,
    solver: PhysicsSolver | None = None,
) -> PipelineContext:
    """Load the protocol once and construct the one allowed physics runtime."""

    root = Path(repository_root).resolve()
    registry = profile_registry or load_pipeline_profile_registry(root)
    if registry.repository_root != root:
        raise Geometry19PipelineContextError(
            "Pipeline profile registry repository root differs from the context root"
        )
    profile = registry.resolve(pipeline_profile)
    source = profile.protocol_path
    contract = load_wp19_23_contract(source)
    protocol = contract.payload
    shadow = protocol.get("shadow")
    delivery = protocol.get("delivery")
    physics_evaluation = protocol.get("physics_evaluation")
    if profile.mode == "paired_physics_shadow":
        valid_shadow = (
            isinstance(shadow, Mapping)
            and shadow.get("enabled") is True
            and shadow.get("delivery_enabled") is False
            and shadow.get("formal_use_allowed") is False
            and shadow.get("forbidden_stages") == ["D", "E", "F"]
            and isinstance(delivery, Mapping)
            and delivery.get("enabled") is False
            and isinstance(physics_evaluation, Mapping)
            and physics_evaluation.get("formal_use_allowed") is False
        )
        if not valid_shadow:
            raise Geometry19PipelineContextError(
                "Paired-shadow profile safety contract is incomplete"
            )
    elif isinstance(shadow, Mapping) and shadow.get("enabled") is True:
        raise Geometry19PipelineContextError(
            "Production-development profile cannot enable paired shadow"
        )
    physics_runtime = build_physics_runtime(root, contract.payload, solver=solver)
    dominance_path = (root / contract.payload["contracts"]["constraint_dominance"]).resolve()
    dominance = ConstraintDominanceContract.from_yaml(dominance_path)
    if dominance.contract_version != contract.payload["optimizer"]["constraint_dominance"]:
        raise Geometry19PipelineContextError(
            "Optimizer and constraint-dominance contract versions differ"
        )
    target_domain_path = contract.payload.get("contracts", {}).get("target_domain")
    target_domain = (
        load_geometry19_target_domain(root, target_domain_path)
        if target_domain_path is not None
        else None
    )
    if profile.name == "v2_2_development" and target_domain is None:
        raise Geometry19PipelineContextError(
            "V2.2 development profile requires a target-domain contract"
        )
    if profile.name != "v2_2_development" and target_domain is not None:
        raise Geometry19PipelineContextError(
            "Target-domain preflight is only allowed for v2_2_development"
        )
    provenance = {
        "pipeline_profile_hash": profile.profile_hash,
        "protocol_hash": _sha256_file(source),
        "constraint_dominance_hash": dominance.source_hash,
        "physics_bundle_hash": physics_runtime.physics_bundle_hash,
        **dict(physics_runtime.provenance_hashes),
    }
    if target_domain is not None:
        provenance["target_domain_hash"] = target_domain.source_hash
    if physics_runtime.model_version != "geometry19_closed_v2_1":
        provenance["pipeline_policy_code_hash"] = _canonical_hash(
            {
                relative: _sha256_file(root / relative)
                for relative in (
                    "src/ai/geometry19_pipeline_context.py",
                    "src/ai/geometry19_pipeline_profiles.py",
                    "src/ai/geometry19_inverse_contract.py",
                    "src/ai/geometry19_pipeline.py",
                    "src/ai/geometry19_selection.py",
                    "src/ai/geometry19_physics.py",
                    "src/ai/geometry19_delivery.py",
                    "src/ai/geometry19_target_domain.py",
                )
            }
        )
    for index, dependency in enumerate(
        contract.payload.get("_protocol_dependencies", ())
    ):
        provenance[f"protocol_dependency_{index}_hash"] = _sha256_file(
            Path(dependency)
        )
    return PipelineContext(
        repository_root=root,
        profile=profile,
        profile_registry_version=registry.registry_version,
        contract_path=source,
        contract=contract,
        physics_runtime=physics_runtime,
        constraint_dominance=dominance,
        target_domain=target_domain,
        provenance_hashes=MappingProxyType(provenance),
        context_hash=_canonical_hash(provenance),
    )


def resolve_pipeline_context(
    repository_root: str | Path,
    pipeline_context: PipelineContext | None,
    *,
    pipeline_profile: str | None = None,
    solver: PhysicsSolver | None = None,
) -> PipelineContext:
    """Return a supplied context or build the default one for a standalone stage."""

    root = Path(repository_root).resolve()
    if pipeline_context is None:
        return build_pipeline_context(
            root, pipeline_profile=pipeline_profile, solver=solver
        )
    if pipeline_context.repository_root != root:
        raise Geometry19PipelineContextError(
            "Pipeline context repository root differs from the stage repository root"
        )
    if solver is not None:
        raise Geometry19PipelineContextError(
            "A stage cannot override the solver of an existing pipeline context"
        )
    if (
        pipeline_profile is not None
        and pipeline_context.profile.name != pipeline_profile
    ):
        raise Geometry19PipelineContextError(
            "Requested pipeline profile differs from the supplied pipeline context"
        )
    return pipeline_context
