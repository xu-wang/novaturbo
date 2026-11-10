"""Public high-precision inverse-design delivery facade.

The operational default is CFM-4 generation followed by Diag-CFM
reverse-consistency prefiltering and real-0D acceptance.  Surrogate-gradient
repair and conditional NSGA-II are independently switchable post-processing
stages.  The former CFM-4-only facade remains available as
``inverse_design_delivery_cfm4_legacy`` for frozen baseline reproduction.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from .baseline import sha256_file, write_yaml
from .diag_cfm_strategy import (
    _load_strategy_target,
    load_diag_cfm_strategy_config,
    run_diag_cfm_strategy_target,
)


DEFAULT_CONFIG = Path("config/inverse_design/default.yaml")


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


def load_delivery_config(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """Load the versioned public high-precision delivery configuration."""

    root = (repo_root or Path.cwd()).resolve()
    delivery_path = _resolve(root, config_path)
    delivery = _load_mapping(delivery_path, "inverse-design delivery")
    required = {
        "version",
        "strategy_id",
        "data",
        "model",
        "generation",
        "delivery",
        "gradient",
        "latent_repair",
        "nsga2",
        "runtime",
        "output",
    }
    missing = required.difference(delivery)
    if missing:
        raise ValueError(
            f"inverse-design delivery config is missing keys: {sorted(missing)}"
        )
    return delivery, {"delivery": delivery_path}


def compose_inverse_design_config(
    config_path: Path,
    target_config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    gradient_enabled_override: Optional[bool] = None,
    nsga2_enabled_override: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """Attach one target and apply explicit post-processing overrides."""

    root = (repo_root or Path.cwd()).resolve()
    delivery, sources = load_delivery_config(config_path, repo_root=root)
    target_path = _resolve(root, target_config_path)
    _load_strategy_target(root, {"spec_yaml": str(target_path)})
    composed = copy.deepcopy(delivery)
    composed["target"] = {"spec_yaml": _portable_path(target_path, root)}
    if gradient_enabled_override is not None:
        composed["gradient"]["enabled"] = bool(gradient_enabled_override)
    if nsga2_enabled_override is not None:
        composed["nsga2"]["enabled"] = bool(nsga2_enabled_override)
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
    *,
    gradient_enabled: bool,
    nsga2_enabled: bool,
) -> None:
    config_dir = output_dir / "configuration"
    config_dir.mkdir()
    records = {}
    for name, source in sources.items():
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
            "provenance_version": "inverse_design_configuration_sources_v2",
            "composed_config": "build_config.yaml",
            "sources": records,
            "effective_postprocessing": {
                "surrogate_gradient_enabled": gradient_enabled,
                "conditional_nsga2_enabled": nsga2_enabled,
            },
        },
    )


def run_inverse_design_delivery(
    config_path: Path,
    target_config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    gradient_enabled_override: Optional[bool] = None,
    nsga2_enabled_override: Optional[bool] = None,
) -> Path:
    """Run the default high-precision chain with auditable optional stages."""

    root = (repo_root or Path.cwd()).resolve()
    composed, sources = compose_inverse_design_config(
        config_path,
        target_config_path,
        repo_root=root,
        gradient_enabled_override=gradient_enabled_override,
        nsga2_enabled_override=nsga2_enabled_override,
    )
    output_override = (
        _resolve(root, output_dir_override)
        if output_dir_override is not None
        else None
    )
    with tempfile.TemporaryDirectory(prefix="novaturbo_inverse_design_") as temporary:
        composed_path = Path(temporary) / "composed.yaml"
        composed_path.write_text(
            yaml.safe_dump(composed, sort_keys=False), encoding="utf-8"
        )
        load_diag_cfm_strategy_config(composed_path)
        output = run_diag_cfm_strategy_target(
            composed_path,
            repo_root=root,
            output_dir_override=output_override,
        )
    _record_configuration(
        output,
        sources,
        root,
        gradient_enabled=bool(composed["gradient"].get("enabled", True)),
        nsga2_enabled=bool(composed["nsga2"].get("enabled", False)),
    )
    return output
