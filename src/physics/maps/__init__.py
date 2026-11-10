"""Validated low-order component-map infrastructure."""

from src.physics.maps.base import (
    ComponentMap,
    MapDomainError,
    MapEvaluation,
    MapMetadataError,
    MapPolicy,
    load_component_map,
)
from src.physics.maps.registry import (
    ComponentMapRegistration,
    ComponentMapRegistry,
    ComponentMapRegistryError,
    load_component_map_registry,
)

__all__ = [
    "ComponentMap",
    "ComponentMapRegistration",
    "ComponentMapRegistry",
    "ComponentMapRegistryError",
    "MapDomainError",
    "MapEvaluation",
    "MapMetadataError",
    "MapPolicy",
    "load_component_map",
    "load_component_map_registry",
]
