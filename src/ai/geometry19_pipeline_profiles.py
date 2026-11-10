"""Version-controlled selectors for complete Geometry19 pipeline protocols.

A profile selects one complete protocol.  It never loads or overrides an
individual physics dependency; that remains the responsibility of the single
``build_physics_runtime()`` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


DEFAULT_PROFILE_REGISTRY_PATH = Path(
    "config/physics/geometry19_pipeline_profiles_v1.yaml"
)
ALLOWED_PIPELINE_MODES = frozenset(
    {"production_development", "paired_physics_shadow"}
)


class Geometry19PipelineProfileError(ValueError):
    """Raised when the profile registry or a requested profile is invalid."""


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _repository_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise Geometry19PipelineProfileError(f"{label} must be a non-empty path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Geometry19PipelineProfileError(
            f"{label} escapes the repository root"
        ) from exc
    if not path.is_file():
        raise Geometry19PipelineProfileError(f"{label} does not exist: {path}")
    return path


@dataclass(frozen=True)
class PipelineProfile:
    """One immutable selection of a complete, versioned pipeline protocol."""

    name: str
    mode: str
    protocol_path: Path
    profile_hash: str

    def contract_mapping(self) -> dict[str, str]:
        return {
            "name": self.name,
            "mode": self.mode,
            "protocol_path": str(self.protocol_path),
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class PipelineProfileRegistry:
    """Validated profile names and their complete protocol selections."""

    repository_root: Path
    source_path: Path
    registry_version: str
    default_profile: str
    profiles: Mapping[str, PipelineProfile]

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def resolve(self, name: str | None = None) -> PipelineProfile:
        selected = self.default_profile if name is None else str(name)
        try:
            return self.profiles[selected]
        except KeyError as exc:
            choices = ", ".join(self.profile_names)
            raise Geometry19PipelineProfileError(
                f"Unknown pipeline profile {selected!r}; registered profiles: {choices}"
            ) from exc


def load_pipeline_profile_registry(
    repository_root: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> PipelineProfileRegistry:
    """Load the selector registry without constructing any physics objects."""

    root = Path(repository_root).resolve()
    if registry_path is None:
        source = (root / DEFAULT_PROFILE_REGISTRY_PATH).resolve()
    else:
        candidate = Path(registry_path)
        source = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise Geometry19PipelineProfileError(
            "Profile registry escapes the repository root"
        ) from exc
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Geometry19PipelineProfileError(
            f"Cannot load pipeline profile registry: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise Geometry19PipelineProfileError("Pipeline profile registry must be a mapping")
    if set(payload) != {"registry_version", "default_profile", "profiles"}:
        raise Geometry19PipelineProfileError(
            "Pipeline profile registry must contain only registry_version, "
            "default_profile, and profiles"
        )
    registry_version = payload["registry_version"]
    default_profile = payload["default_profile"]
    raw_profiles = payload["profiles"]
    if not isinstance(registry_version, str) or not registry_version:
        raise Geometry19PipelineProfileError("registry_version must be non-empty")
    if not isinstance(default_profile, str) or not default_profile:
        raise Geometry19PipelineProfileError("default_profile must be non-empty")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise Geometry19PipelineProfileError("profiles must be a non-empty mapping")

    resolved: dict[str, PipelineProfile] = {}
    for name, spec in raw_profiles.items():
        if not isinstance(name, str) or not name:
            raise Geometry19PipelineProfileError("Profile names must be non-empty strings")
        if not isinstance(spec, dict) or set(spec) != {"protocol", "mode"}:
            raise Geometry19PipelineProfileError(
                f"Profile {name!r} must contain only protocol and mode"
            )
        mode = spec["mode"]
        if not isinstance(mode, str) or mode not in ALLOWED_PIPELINE_MODES:
            raise Geometry19PipelineProfileError(
                f"Profile {name!r} has unsupported mode {mode!r}; "
                f"allowed={sorted(ALLOWED_PIPELINE_MODES)}"
            )
        protocol_path = _repository_file(
            root, spec["protocol"], f"profiles.{name}.protocol"
        )
        identity = {
            "registry_version": registry_version,
            "name": name,
            "mode": mode,
            "protocol": str(protocol_path.relative_to(root)),
        }
        resolved[name] = PipelineProfile(
            name=name,
            mode=mode,
            protocol_path=protocol_path,
            profile_hash=_canonical_hash(identity),
        )
    if default_profile not in resolved:
        raise Geometry19PipelineProfileError(
            "default_profile is not present in profiles"
        )
    return PipelineProfileRegistry(
        repository_root=root,
        source_path=source,
        registry_version=registry_version,
        default_profile=default_profile,
        profiles=MappingProxyType(resolved),
    )
