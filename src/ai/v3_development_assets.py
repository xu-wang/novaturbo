"""Deterministic 22D software fixtures for V3 B--F interface development."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import yaml

from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import (
    Geometry22Design,
    Geometry22DesignError,
    SCHEMA_PATH,
    geometry22_schema_hash,
)
from src.physics.v3_development_runtime import (
    MODEL_VERSION,
    V3DevelopmentPhysicsRuntime,
)


ASSET_VERSION = "v3_22d_software_fixture_v1"
FILES = ("cases.jsonl", "generator_asset.json", "forward_asset.json")
OUTPUT_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class V3FixtureAssetError(ValueError):
    """A V3 fixture is missing, incompatible, or altered."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_designs() -> tuple[Geometry22Design, ...]:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = schema["design_variables"]
    reference = {row["name"]: row["reference_value"] for row in fields}
    designs = [Geometry22Design.from_mapping(reference)]
    for row in fields:
        name = row["name"]
        lower, upper = row["bounds"]
        center = row["reference_value"]
        if row["dtype"] == "int64":
            values = (max(lower, center - 1), min(upper, center + 1))
        elif row["index"] >= 20:
            values = (lower, upper)
        else:
            values = (0.5 * (lower + center), 0.5 * (center + upper))
        if values[0] == values[1] or center in values:
            raise V3FixtureAssetError(f"Cannot form two distinct OAT cases for {name}")
        for value in values:
            designs.append(Geometry22Design.from_mapping({**reference, name: value}))
    if len(designs) != 45 or len({design.design_id() for design in designs}) != 45:
        raise V3FixtureAssetError("V3 fixture designs are not 45 unique 22D coordinates")
    return tuple(designs)


def _evaluate_rows(runtime: V3DevelopmentPhysicsRuntime) -> list[dict[str, Any]]:
    normalizer = Geometry22Normalizer()
    if normalizer.schema_hash != runtime.design_schema_hash:
        raise V3FixtureAssetError("V3 fixture normalizer schema differs from runtime")
    rows = []
    for index, design in enumerate(_candidate_designs()):
        result = runtime.evaluate(design)
        engine = result.engine_result
        if engine.solver.converged:
            try:
                performance = {name: float(engine.performance[name]) for name in OUTPUT_NAMES}
            except (KeyError, TypeError, ValueError) as exc:
                raise V3FixtureAssetError("Converged V3 case has incomplete performance") from exc
            if not all(math.isfinite(value) for value in performance.values()):
                raise V3FixtureAssetError("Converged V3 case has non-finite performance")
        else:
            performance = None
        rows.append(
            {
                "case_index": index,
                "case_name": "reference" if index == 0 else f"axis_{(index - 1) // 2 + 1:02d}_{'low' if index % 2 else 'high'}",
                "design_id": result.design_id,
                "design": design.as_mapping(),
                "normalized_design": list(normalizer.transform(design)),
                "geometry19_projection_hash": design.geometry19_projection_hash(),
                "physics_model_version": result.physics_model_version,
                "physics_bundle_hash": result.physics_bundle_hash,
                "solver_converged": engine.solver.converged,
                "solver_failure_code": (
                    engine.solver.failure_code.value if engine.solver.failure_code else None
                ),
                "flow_root_count": engine.operating_point.get("flow_root_count"),
                "performance": performance,
                "physics_validation_status": "not_completed",
                "formal_use_allowed": False,
                "deployment_allowed": False,
            }
        )
    return rows


def _asset_files(runtime: V3DevelopmentPhysicsRuntime) -> dict[str, bytes]:
    rows = _evaluate_rows(runtime)
    cases = b"".join(_json_bytes(row) for row in rows)
    common = {
        "asset_version": ASSET_VERSION,
        "design_schema_hash": runtime.design_schema_hash,
        "physics_bundle_hash": runtime.physics_bundle_hash,
        "physics_model_version": MODEL_VERSION,
        "source_cases_sha256": hashlib.sha256(cases).hexdigest(),
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    generator = {
        **common,
        "asset_kind": "deterministic_22d_candidate_table_fixture",
        "input_dimension": 22,
        "target_conditioned": False,
        "training_status": "not_trained",
        "candidate_design_ids": [row["design_id"] for row in rows],
    }
    forward = {
        **common,
        "asset_kind": "exact_label_lookup_fixture",
        "input_dimension": 22,
        "output_names": list(OUTPUT_NAMES),
        "training_status": "not_trained",
        "prediction_scope": "only_listed_converged_design_ids",
        "labeled_design_ids": [
            row["design_id"] for row in rows if row["performance"] is not None
        ],
    }
    return {
        "cases.jsonl": cases,
        "generator_asset.json": _json_bytes(generator),
        "forward_asset.json": _json_bytes(forward),
    }


def build_v3_fixture_assets(
    output_root: str | Path,
    runtime: V3DevelopmentPhysicsRuntime,
) -> dict[str, Any]:
    """Create immutable, isolated software assets; never overwrite an existing run."""

    output = Path(output_root).resolve()
    if output.exists():
        raise V3FixtureAssetError(f"V3 fixture output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_22d_fixture_", dir=output.parent))
    try:
        files = _asset_files(runtime)
        for name, content in files.items():
            (temporary / name).write_bytes(content)
        rows = [json.loads(line) for line in files["cases.jsonl"].splitlines()]
        manifest = {
            "asset_version": ASSET_VERSION,
            "asset_status": "software_fixture_only_unvalidated",
            "design_dimension": 22,
            "design_schema_hash": runtime.design_schema_hash,
            "physics_bundle_hash": runtime.physics_bundle_hash,
            "physics_model_version": MODEL_VERSION,
            "runtime_contract": runtime.contract_mapping(),
            "normalization": "physical_bounds_to_minus_one_plus_one_v1",
            "normalizer_schema_hash": runtime.design_schema_hash,
            "split": "software_fixture_only_no_train_validation_claim",
            "row_count": len(rows),
            "converged_count": sum(row["performance"] is not None for row in rows),
            "failure_count": sum(row["performance"] is None for row in rows),
            "unique_design_count": len({row["design_id"] for row in rows}),
            "axis_oat_coverage": 22,
            "joint_coverage": "not_established",
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
            "physics_validation_status": "not_completed",
            "model_effectiveness_status": "not_evaluated",
            "formal_use_allowed": False,
            "deployment_allowed": False,
        }
        (temporary / "complete_manifest.json").write_bytes(_json_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_v3_fixture_assets(output, runtime).manifest


@dataclass(frozen=True)
class V3FixtureAssets:
    manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    generator: Mapping[str, Any]
    forward: Mapping[str, Any]

    def candidate_designs(self, limit: int | None = None) -> tuple[Geometry22Design, ...]:
        if limit is not None and (isinstance(limit, bool) or limit < 0):
            raise V3FixtureAssetError("Candidate limit must be nonnegative")
        selected = self.rows if limit is None else self.rows[:limit]
        return tuple(Geometry22Design.from_mapping(row["design"]) for row in selected)

    def predict_known(self, design: Geometry22Design) -> dict[str, Any]:
        if not isinstance(design, Geometry22Design):
            raise V3FixtureAssetError("Forward fixture requires a 22D design")
        row = next((item for item in self.rows if item["design_id"] == design.design_id()), None)
        if row is None or row["performance"] is None:
            raise V3FixtureAssetError("No converged fixture label for this 22D design")
        return {
            "prediction_kind": "exact_stored_label_fixture_not_general_surrogate",
            "design_id": design.design_id(),
            "physics_bundle_hash": self.manifest["physics_bundle_hash"],
            "performance": dict(row["performance"]),
            "formal_use_allowed": False,
        }


def load_v3_fixture_assets(
    output_root: str | Path,
    runtime: V3DevelopmentPhysicsRuntime,
) -> V3FixtureAssets:
    root = Path(output_root).resolve()
    try:
        manifest = json.loads((root / "complete_manifest.json").read_text(encoding="utf-8"))
        for name in FILES:
            if _file_hash(root / name) != manifest["files"][name]:
                raise V3FixtureAssetError(f"V3 fixture file changed: {name}")
        rows = tuple(json.loads(line) for line in (root / "cases.jsonl").read_text(encoding="utf-8").splitlines())
        generator = json.loads((root / "generator_asset.json").read_text(encoding="utf-8"))
        forward = json.loads((root / "forward_asset.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3FixtureAssetError("Cannot load complete V3 fixture assets") from exc
    if (
        manifest.get("asset_version") != ASSET_VERSION
        or manifest.get("design_schema_hash") != geometry22_schema_hash()
        or manifest.get("design_schema_hash") != runtime.design_schema_hash
        or manifest.get("physics_bundle_hash") != runtime.physics_bundle_hash
        or manifest.get("physics_model_version") != MODEL_VERSION
        or manifest.get("runtime_contract") != runtime.contract_mapping()
        or manifest.get("asset_status") != "software_fixture_only_unvalidated"
        or manifest.get("row_count") != len(rows)
        or manifest.get("unique_design_count") != len({row["design_id"] for row in rows})
        or manifest.get("row_count") != 45
        or manifest.get("converged_count") != sum(row["performance"] is not None for row in rows)
        or manifest.get("failure_count") != sum(row["performance"] is None for row in rows)
        or manifest.get("normalizer_schema_hash") != runtime.design_schema_hash
        or manifest.get("axis_oat_coverage") != 22
        or manifest.get("joint_coverage") != "not_established"
        or manifest.get("formal_use_allowed") is not False
        or manifest.get("deployment_allowed") is not False
        or generator.get("asset_kind") != "deterministic_22d_candidate_table_fixture"
        or forward.get("asset_kind") != "exact_label_lookup_fixture"
        or generator.get("source_cases_sha256") != manifest["files"]["cases.jsonl"]
        or forward.get("source_cases_sha256") != manifest["files"]["cases.jsonl"]
        or generator.get("design_schema_hash") != runtime.design_schema_hash
        or forward.get("design_schema_hash") != runtime.design_schema_hash
        or generator.get("physics_bundle_hash") != runtime.physics_bundle_hash
        or forward.get("physics_bundle_hash") != runtime.physics_bundle_hash
        or generator.get("candidate_design_ids") != [row["design_id"] for row in rows]
        or forward.get("labeled_design_ids") != [row["design_id"] for row in rows if row["performance"] is not None]
    ):
        raise V3FixtureAssetError("V3 fixture identity or content mismatch")
    normalizer = Geometry22Normalizer()
    for index, row in enumerate(rows):
        try:
            design = Geometry22Design.from_mapping(row["design"])
        except (KeyError, Geometry22DesignError) as exc:
            raise V3FixtureAssetError("V3 fixture contains invalid 22D design") from exc
        if (
            row["case_index"] != index
            or row["design_id"] != design.design_id()
            or row["normalized_design"] != list(normalizer.transform(design))
            or row["geometry19_projection_hash"] != design.geometry19_projection_hash()
            or row["physics_bundle_hash"] != runtime.physics_bundle_hash
            or row["physics_model_version"] != MODEL_VERSION
            or row["formal_use_allowed"] is not False
            or (row["performance"] is None) != (not row["solver_converged"])
        ):
            raise V3FixtureAssetError("V3 fixture row identity or status mismatch")
        if row["performance"] is not None and not all(
            math.isfinite(float(row["performance"][name])) for name in OUTPUT_NAMES
        ):
            raise V3FixtureAssetError("V3 fixture row has non-finite label")
    return V3FixtureAssets(manifest, rows, generator, forward)
