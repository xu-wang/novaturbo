"""Repository-scoped registry for versioned low-order component maps."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from src.physics.maps.base import ComponentMap, MapMetadataError, load_component_map


COMPONENT_TYPES = frozenset({"compressor", "turbine", "combustor", "nozzle", "inlet"})
USAGE_LEVELS = frozenset(
    {"software_fixture", "validation_shadow", "engineering_candidate", "formal"}
)


class ComponentMapRegistryError(ValueError):
    """Raised when map registration or selection violates the frozen contract."""


@dataclass(frozen=True)
class ComponentMapRegistration:
    """One validated registry entry and its immutable loaded map."""

    component_type: str
    map_id: str
    map_version: str
    source_hash: str
    usage_level: str
    formal_use_allowed: bool
    package_path: Path
    component_map: ComponentMap


@dataclass(frozen=True)
class ComponentMapRegistry:
    """Validated repository registry; protocols select IDs, never package paths."""

    repository_root: Path
    source_path: Path
    registry_version: str
    source_hash: str
    registrations: Mapping[str, ComponentMapRegistration]

    def resolve(
        self, selections: Mapping[str, Any]
    ) -> Mapping[str, ComponentMap]:
        """Resolve an exact component-to-map-ID selection."""

        if not isinstance(selections, Mapping) or not selections:
            raise ComponentMapRegistryError(
                "Component-map selection must be a non-empty mapping"
            )
        resolved: dict[str, ComponentMap] = {}
        for component_type, map_id in selections.items():
            if component_type not in COMPONENT_TYPES:
                raise ComponentMapRegistryError(
                    f"Unsupported component-map selection: {component_type!r}"
                )
            if not isinstance(map_id, str) or not map_id:
                raise ComponentMapRegistryError(
                    f"Map selection for {component_type} must be a non-empty ID"
                )
            try:
                registration = self.registrations[map_id]
            except KeyError as exc:
                raise ComponentMapRegistryError(
                    f"Unregistered component map: {map_id!r}"
                ) from exc
            if registration.component_type != component_type:
                raise ComponentMapRegistryError(
                    f"Map {map_id!r} is registered for "
                    f"{registration.component_type}, not {component_type}"
                )
            resolved[component_type] = registration.component_map
        return MappingProxyType(resolved)

    def registration(
        self, component_type: str, map_id: str
    ) -> ComponentMapRegistration:
        """Return the selected entry after checking its component ownership."""

        self.resolve({component_type: map_id})
        return self.registrations[map_id]


def _repository_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ComponentMapRegistryError(f"{label} must be a non-empty path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ComponentMapRegistryError(f"{label} escapes the repository root") from exc
    return path


def load_component_map_registry(
    repository_root: str | Path,
    registry_path: str | Path,
) -> ComponentMapRegistry:
    """Load all registered packages and validate registry-to-metadata identity."""

    root = Path(repository_root).resolve()
    candidate = Path(registry_path)
    source = (
        candidate.resolve()
        if candidate.is_absolute()
        else _repository_path(root, str(candidate), "component-map registry")
    )
    try:
        source_bytes = source.read_bytes()
        payload = yaml.safe_load(source_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise ComponentMapRegistryError(
            f"Cannot load component-map registry: {source}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "registry_version",
        "maps",
    }:
        raise ComponentMapRegistryError("Component-map registry keys mismatch")
    registry_version = payload["registry_version"]
    if not isinstance(registry_version, str) or not registry_version:
        raise ComponentMapRegistryError("registry_version must be non-empty")
    if registry_version != source.stem:
        raise ComponentMapRegistryError("Registry filename/version mismatch")
    entries = payload["maps"]
    if not isinstance(entries, list) or not entries:
        raise ComponentMapRegistryError("Component-map registry must contain maps")

    required_entry_keys = {
        "component_type",
        "map_id",
        "map_version",
        "source_hash",
        "usage_level",
        "formal_use_allowed",
        "package",
    }
    registrations: dict[str, ComponentMapRegistration] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != required_entry_keys:
            raise ComponentMapRegistryError(
                f"Component-map registry entry {index} keys mismatch"
            )
        component_type = entry["component_type"]
        map_id = entry["map_id"]
        map_version = entry["map_version"]
        source_hash = entry["source_hash"]
        usage_level = entry["usage_level"]
        if not isinstance(component_type, str) or component_type not in COMPONENT_TYPES:
            raise ComponentMapRegistryError(
                f"Unsupported registered component type: {component_type!r}"
            )
        if not isinstance(map_id, str) or not map_id:
            raise ComponentMapRegistryError("Registered map_id must be non-empty")
        if map_id in registrations:
            raise ComponentMapRegistryError(f"Duplicate registered map ID: {map_id}")
        if not isinstance(map_version, str) or not map_version:
            raise ComponentMapRegistryError("Registered map_version must be non-empty")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ComponentMapRegistryError(
                "Registered source_hash must be lowercase SHA-256"
            )
        if not isinstance(usage_level, str) or usage_level not in USAGE_LEVELS:
            raise ComponentMapRegistryError(
                f"Unsupported map usage level: {usage_level!r}"
            )
        formal_use_allowed = entry["formal_use_allowed"]
        if type(formal_use_allowed) is not bool:
            raise ComponentMapRegistryError("formal_use_allowed must be boolean")
        if (usage_level == "formal") != formal_use_allowed:
            raise ComponentMapRegistryError(
                "Only formal registry entries may allow formal map use"
            )
        package_path = _repository_path(
            root, entry["package"], f"maps[{index}].package"
        )
        if not package_path.is_dir():
            raise ComponentMapRegistryError(
                f"Registered map package does not exist: {package_path}"
            )
        try:
            component_map = load_component_map(package_path)
        except MapMetadataError as exc:
            raise ComponentMapRegistryError(
                f"Registered map package is invalid: {map_id!r}"
            ) from exc
        expected_identity = (
            component_type,
            map_id,
            map_version,
            source_hash,
            formal_use_allowed,
        )
        actual_identity = (
            component_map.component_type,
            component_map.map_id,
            component_map.map_version,
            component_map.source_hash,
            component_map.formal_use_allowed,
        )
        if actual_identity != expected_identity:
            raise ComponentMapRegistryError(
                f"Registry/package identity mismatch for {map_id!r}"
            )
        registrations[map_id] = ComponentMapRegistration(
            component_type=component_type,
            map_id=map_id,
            map_version=map_version,
            source_hash=source_hash,
            usage_level=usage_level,
            formal_use_allowed=formal_use_allowed,
            package_path=package_path,
            component_map=component_map,
        )
    return ComponentMapRegistry(
        repository_root=root,
        source_path=source,
        registry_version=registry_version,
        source_hash=sha256(source_bytes).hexdigest(),
        registrations=MappingProxyType(registrations),
    )
