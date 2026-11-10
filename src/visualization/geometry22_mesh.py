"""Publish traceable V3 visualization meshes for complete 22D designs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from src.physics.geometry22_design import Geometry22Design
from src.physics.model_config import ModelConfig
from src.visualization.geometry19_adapter import (
    _canonical_hash,
    _solid_mapping,
    geometry19_to_assembly_params,
)
from src.visualization.geometry19_mesh import generate_geometry19_meshes


class Geometry22MeshError(ValueError):
    """Invalid V3 mesh identity, publication, or cached artifact."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Geometry22MeshStore:
    def __init__(self, repository_root: str | Path, storage_root: str | Path):
        self.repository = Path(repository_root).resolve()
        self.storage = Path(storage_root).resolve()

    def _sources(self) -> dict[str, str]:
        names = {
            "config/physics/geometry19_calibration_bootstrap_v1.yaml",
            "config/physics/geometry19_visualization_calibration_v1.yaml",
            "config/physics/geometry19_mesh_profiles_v1.yaml",
            "config/physics/geometry22_schema_v3_development_v1.yaml",
            "src/physics/geometry22_design.py",
            "src/physics/geometry_state.py",
            "src/physics/mass_model_v2.py",
            "src/physics/model_config.py",
            "src/visualization/geometry19_adapter.py",
            "src/visualization/geometry19_mesh.py",
            "src/visualization/geometry22_mesh.py",
            "src/export/stl_export.py",
        }
        names.update(str(path.relative_to(self.repository)) for path in (self.repository / "src/geometry").rglob("*.py"))
        return {name: _sha(self.repository / name) for name in sorted(names)}

    def _identity(self, design: Geometry22Design, quality: str) -> dict:
        if quality not in {"preview_mesh", "export_mesh"}:
            raise Geometry22MeshError("Unsupported V3 mesh quality")
        return {"version": "geometry22_visual_mesh_v1", "design_id": design.design_id(),
                "quality": quality, "sources": self._sources()}

    @staticmethod
    def assembly(design: Geometry22Design, repository_root: str | Path):
        """Map geometric V3 coordinates to solids; retain air split as flow metadata."""
        root = Path(repository_root)
        config = ModelConfig.from_yaml(root / "config/physics/geometry19_calibration_bootstrap_v1.yaml")
        config = config.with_overrides(parameter_values={
            "diffuser_area_ratio": design.component_parameters()["diffuser_area_ratio"],
            "combustor_primary_air_fraction": design.component_parameters()["combustor_primary_air_fraction"],
        })
        base = geometry19_to_assembly_params(design.geometry19_projection(), config)
        gap_mm = (design.component_parameters()["compressor_tip_clearance_ratio"]
                  * design.as_mapping()["compressor_impeller_exit_diameter_m"] * 1000.0)
        compressor = replace(base.assembly_params.compressor, shroud_clearance_mm=gap_mm)
        params = replace(base.assembly_params, compressor=compressor)
        solid_hash = _canonical_hash(_solid_mapping(params))
        lattice_hash = _canonical_hash({"solid_geometry_hash": solid_hash, "lattice": base.lattice_parameters})
        return replace(base, assembly_params=params, solid_geometry_hash=solid_hash,
                       lattice_hash=lattice_hash, design_hash=design.design_id(),
                       overlay_parameters={**base.overlay_parameters,
                                           "combustor_primary_air_fraction": design.component_parameters()["combustor_primary_air_fraction"]})

    def publish(self, design: Geometry22Design, quality: str = "preview_mesh") -> tuple[dict, Path]:
        identity = self._identity(design, quality)
        mesh_id = _canonical(identity)
        directory = self.storage / mesh_id
        if not (directory / "complete_manifest.json").is_file():
            self.storage.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".v3_mesh_", dir=self.storage))
            try:
                assembly = self.assembly(design, self.repository)
                result = generate_geometry19_meshes(assembly, temporary, quality)
                if not result.passed:
                    raise Geometry22MeshError("V3 mesh failed mandatory quality diagnostics")
                components = {path.stem: {"filename": path.name, "sha256": _sha(path),
                                          "size_bytes": path.stat().st_size}
                              for path in sorted(temporary.glob("*.stl"))}
                if not components:
                    raise Geometry22MeshError("V3 mesh contains no STL components")
                manifest = {"identity": identity, "mesh_id": mesh_id, "geometry22": design.as_mapping(),
                            "components": components, "solid_geometry_hash": assembly.solid_geometry_hash,
                            "lattice_hash": assembly.lattice_hash, "calibration_hash": assembly.calibration_hash,
                            "mesh_diagnostics": result.as_mapping(),
                            "geometric_v3_parameters": ["diffuser_area_ratio", "compressor_tip_clearance_ratio"],
                            "flow_only_parameters": {"combustor_primary_air_fraction": design.component_parameters()["combustor_primary_air_fraction"]},
                            "formal_use_allowed": False, "deployment_allowed": False}
                (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
                files = {path.name: _sha(path) for path in temporary.iterdir() if path.is_file()}
                (temporary / "complete_manifest.json").write_text(json.dumps({"files": files}, sort_keys=True) + "\n")
                try:
                    os.rename(temporary, directory)
                except OSError:
                    if not (directory / "complete_manifest.json").is_file():
                        raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return self.get(mesh_id, design=design, quality=quality)

    def get(self, mesh_id: str, *, design: Geometry22Design | None = None,
            quality: str | None = None) -> tuple[dict, Path]:
        if not isinstance(mesh_id, str) or not re.fullmatch(r"[0-9a-f]{64}", mesh_id):
            raise Geometry22MeshError("Invalid V3 mesh ID")
        directory = self.storage / mesh_id
        if not (directory / "complete_manifest.json").is_file():
            raise FileNotFoundError("V3 mesh not found")
        seal = json.loads((directory / "complete_manifest.json").read_text())
        actual = {path.name for path in directory.iterdir() if path.is_file() and path.name != "complete_manifest.json"}
        if set(seal["files"]) != actual or any(
            Path(name).name != name or _sha(directory / name) != digest
            for name, digest in seal["files"].items()
        ):
            raise Geometry22MeshError("V3 mesh artifact changed")
        manifest = json.loads((directory / "manifest.json").read_text())
        recorded = Geometry22Design.from_mapping(manifest["geometry22"])
        expected = self._identity(recorded, manifest["identity"]["quality"])
        if (manifest["identity"] != expected or manifest["mesh_id"] != mesh_id
                or _canonical(expected) != mesh_id or recorded.design_id() != expected["design_id"]
                or (design is not None and design != recorded)
                or (quality is not None and quality != expected["quality"])
                or manifest["formal_use_allowed"] is not False
                or manifest["deployment_allowed"] is not False
                or manifest["mesh_diagnostics"]["passed"] is not True):
            raise Geometry22MeshError("V3 mesh identity changed")
        for component in manifest["components"].values():
            name = component["filename"]
            if Path(name).name != name or _sha(directory / name) != component["sha256"]:
                raise Geometry22MeshError("V3 mesh component changed")
        return manifest, directory

    def response(self, manifest: dict, url_prefix: str) -> dict:
        return {"mesh_id": manifest["mesh_id"], "design_id": manifest["identity"]["design_id"],
                "geometry22": manifest["geometry22"], "mesh_quality": manifest["identity"]["quality"],
                "mesh_scope": "22d_design_driven_visual_mesh", "mesh_diagnostics": manifest["mesh_diagnostics"],
                "geometry_hash": manifest["solid_geometry_hash"], "lattice_hash": manifest["lattice_hash"],
                "geometric_v3_parameters": manifest["geometric_v3_parameters"],
                "flow_only_parameters": manifest["flow_only_parameters"],
                "component_urls": {name: f"{url_prefix}/{manifest['mesh_id']}/{item['filename']}?h={item['sha256']}"
                                   for name, item in manifest["components"].items()},
                "formal_use_allowed": False, "deployment_allowed": False}
