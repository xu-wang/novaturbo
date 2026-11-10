"""Legacy frozen CFM-4 inverse-design delivery facade.

This module composes the public policy, generator, and target configurations
into the frozen Stage-II WP5 execution schema.  The numerical implementation
remains in :mod:`src.ai.stage2_wp5` so the maintenance facade cannot silently
diverge from the validated delivery chain.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from .baseline import sha256_file, write_yaml
from .inverse_design import TargetSpec
from .stage2_wp5 import load_stage2_wp5_config, run_stage2_wp5


DEFAULT_CONFIG = Path("config/inverse_design/legacy/cfm4_default_v1.yaml")


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_mapping(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} config is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} config must contain a YAML mapping: {path}")
    return dict(value)


def _require_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{label} config is missing keys: {sorted(missing)}")


def load_delivery_config(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """Load and validate the public delivery configuration and its components."""

    root = (repo_root or Path.cwd()).resolve()
    delivery_path = _resolve(root, config_path)
    delivery = _load_mapping(delivery_path, "inverse-design delivery")
    _require_keys(
        delivery,
        {
            "version",
            "composed_version",
            "policy_config",
            "generator_config",
            "generation",
            "runtime",
            "output",
        },
        "inverse-design delivery",
    )
    policy_path = _resolve(root, str(delivery["policy_config"]))
    generator_path = _resolve(root, str(delivery["generator_config"]))
    policy = _load_mapping(policy_path, "inverse-design policy")
    generator = _load_mapping(generator_path, "inverse-design generator")
    _require_keys(
        policy,
        {
            "policy_version",
            "data",
            "generation",
            "delivery",
            "gradient",
            "latent_repair",
            "nsga2",
        },
        "inverse-design policy",
    )
    _require_keys(generator, {"generator_version", "model"}, "inverse-design generator")
    delivery["_policy"] = policy
    delivery["_generator"] = generator
    return delivery, {
        "delivery": delivery_path,
        "policy": policy_path,
        "generator": generator_path,
    }


def compose_inverse_design_config(
    config_path: Path,
    target_config_path: Path,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """Compose the stable public configs into the unchanged WP5 run schema."""

    root = (repo_root or Path.cwd()).resolve()
    delivery, sources = load_delivery_config(config_path, repo_root=root)
    target_path = _resolve(root, target_config_path)
    target_mapping = _load_mapping(target_path, "inverse-design target")
    TargetSpec.from_mapping(target_mapping)
    policy = delivery.pop("_policy")
    generator = delivery.pop("_generator")

    generation = copy.deepcopy(policy["generation"])
    generation.update(copy.deepcopy(delivery["generation"]))
    composed = {
        "version": str(delivery["composed_version"]),
        "data": copy.deepcopy(policy["data"]),
        "target": {"spec_yaml": _portable_path(target_path, root)},
        "model": copy.deepcopy(generator["model"]),
        "generation": generation,
        "delivery": copy.deepcopy(policy["delivery"]),
        "gradient": copy.deepcopy(policy["gradient"]),
        "latent_repair": copy.deepcopy(policy["latent_repair"]),
        "nsga2": copy.deepcopy(policy["nsga2"]),
        "runtime": copy.deepcopy(delivery["runtime"]),
        "output": copy.deepcopy(delivery["output"]),
    }
    sources["target"] = target_path
    return composed, sources


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _record_configuration(
    output_dir: Path,
    sources: Mapping[str, Path],
    repo_root: Path,
) -> None:
    config_dir = output_dir / "configuration"
    config_dir.mkdir()
    records = {}
    for name in ("delivery", "policy", "generator", "target"):
        source = sources[name]
        copied_name = f"{name}.yaml"
        shutil.copy2(source, config_dir / copied_name)
        records[name] = {
            "source": _portable_path(source, repo_root),
            "sha256": sha256_file(source),
            "copy": f"configuration/{copied_name}",
        }
    write_yaml(
        config_dir / "sources.yaml",
        {
            "provenance_version": "inverse_design_configuration_sources_v1",
            "composed_config": "build_config.yaml",
            "sources": records,
        },
    )


def run_inverse_design_delivery(
    config_path: Path,
    target_config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    """Run the stable inverse-design entry while preserving WP5 numerics."""

    root = (repo_root or Path.cwd()).resolve()
    composed, sources = compose_inverse_design_config(
        config_path,
        target_config_path,
        repo_root=root,
    )
    output_override = (
        _resolve(root, output_dir_override)
        if output_dir_override is not None
        else None
    )
    with tempfile.TemporaryDirectory(prefix="novaturbo_inverse_design_") as temporary:
        composed_path = Path(temporary) / "composed.yaml"
        composed_path.write_text(
            yaml.safe_dump(composed, sort_keys=False),
            encoding="utf-8",
        )
        # Validate before creating the output directory, then execute the same
        # frozen numerical core used by the legacy entry point.
        load_stage2_wp5_config(composed_path)
        output = run_stage2_wp5(
            composed_path,
            repo_root=root,
            output_dir_override=output_override,
        )
    _record_configuration(output, sources, root)
    return output
