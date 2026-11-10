"""Tests for the WP1 generator_v1 dataset construction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ai.dataset import DesignSpaceBounds, generate_dataset
from src.ai.generator_dataset import (
    GeneratorNormalizer,
    assign_splits,
    build_generator_views,
    design_names,
    engine_inputs_from_cycle_row,
    load_schema,
    run_parity_check,
    run_wp1,
    stable_sample_id,
    validate_schema_against_design_bounds,
)
from src.physics.brayton import FlightConditions, solve_brayton_cycle


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "generator_v1_schema.yaml"


def _schema():
    return load_schema(SCHEMA_PATH)


def _source_frame(n_rows=20):
    bounds = DesignSpaceBounds()
    rows = []
    for index in range(n_rows):
        fraction = (index + 0.5) / n_rows
        row = {
            name: lower + fraction * (upper - lower)
            for name, (lower, upper) in (
                (field, getattr(bounds, field))
                for field in bounds.__dataclass_fields__
            )
        }
        row.update(
            {
                "thrust_N": 40.0 + index,
                "tsfc_kg_N_s": 3.0e-5 + index * 1.0e-7,
                "is_valid": True,
                "n_warnings": 0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_schema_bounds_match_design_space():
    schema = _schema()
    validate_schema_against_design_bounds(schema)
    assert design_names(schema) == [
        "compressor_pressure_ratio",
        "compressor_efficiency",
        "turbine_inlet_temp_K",
        "turbine_efficiency",
        "mass_flow_kg_s",
    ]


def test_stable_sample_id_is_reproducible_and_content_derived():
    names = design_names(_schema())
    values = {name: float(index + 1) for index, name in enumerate(names)}
    first = stable_sample_id("v1", "source.csv", 3, values, names)
    second = stable_sample_id("v1", "source.csv", 3, values, names)
    changed_row = stable_sample_id("v1", "source.csv", 4, values, names)
    changed_values = dict(values)
    changed_values[names[0]] += 0.01
    changed_design = stable_sample_id(
        "v1", "source.csv", 3, changed_values, names
    )
    assert first == second
    assert len(first) == 64
    assert first != changed_row
    assert first != changed_design


def test_generator_views_keep_invalid_rows():
    source = _source_frame(5)
    source.loc[2, "is_valid"] = False
    source.loc[2, "thrust_N"] = np.nan
    source.loc[2, "tsfc_kg_N_s"] = np.inf
    all_view, valid_view, invalid_view = build_generator_views(
        source, _schema(), "test_v1", "source.csv"
    )
    assert len(all_view) == 5
    assert len(valid_view) == 4
    assert len(invalid_view) == 1
    assert invalid_view.iloc[0]["failure_code"] == "SOURCE_INVALID"
    assert invalid_view.iloc[0]["split"] == "invalid"


def test_assign_splits_is_deterministic_and_disjoint():
    all_view, valid_view, _ = build_generator_views(
        _source_frame(30), _schema(), "test_v1", "source.csv"
    )
    first_all, first_valid, first_manifest = assign_splits(
        all_view, valid_view, 42, 0.8, 0.1, 0.1
    )
    second_all, second_valid, second_manifest = assign_splits(
        all_view, valid_view, 42, 0.8, 0.1, 0.1
    )
    pd.testing.assert_frame_equal(first_all, second_all)
    pd.testing.assert_frame_equal(first_valid, second_valid)
    pd.testing.assert_frame_equal(first_manifest, second_manifest)
    assert first_valid["split"].value_counts().to_dict() == {
        "train": 24,
        "val": 3,
        "test": 3,
    }
    sets = {
        split: set(group["sample_id"])
        for split, group in first_manifest.groupby("split")
    }
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_normalizer_uses_train_rows_and_round_trips():
    all_view, valid_view, _ = build_generator_views(
        _source_frame(20), _schema(), "test_v1", "source.csv"
    )
    _, valid_view, _ = assign_splits(
        all_view, valid_view, 42, 0.8, 0.1, 0.1
    )
    train = valid_view[valid_view["split"] == "train"]
    normalizer = GeneratorNormalizer.fit(train, _schema())

    expected_thrust_mean = train["thrust_N"].mean()
    thrust_spec = next(
        item for item in normalizer.conditions if item["name"] == "thrust_N"
    )
    assert thrust_spec["mean"] == pytest.approx(expected_thrust_mean)
    assert normalizer.fitted_sample_count == len(train)

    design = valid_view.iloc[:5][normalizer.design_names]
    design_restored = normalizer.inverse_design(
        normalizer.transform_design(design)
    )
    np.testing.assert_allclose(design_restored, design.to_numpy(), atol=1e-12)

    conditions = valid_view.iloc[:5][normalizer.condition_names]
    condition_restored = normalizer.inverse_conditions(
        normalizer.transform_conditions(conditions)
    )
    np.testing.assert_allclose(
        condition_restored, conditions.to_numpy(), rtol=1e-12, atol=1e-12
    )


def test_cycle_field_mapping_reproduces_solver_output():
    schema = _schema()
    row = {
        "compressor_pressure_ratio": 3.5,
        "compressor_efficiency": 0.78,
        "turbine_inlet_temp_K": 1100.0,
        "turbine_efficiency": 0.82,
        "mass_flow_kg_s": 0.15,
    }
    engine = engine_inputs_from_cycle_row(row, schema)
    assert engine.compressor_isentropic_efficiency == pytest.approx(0.78)
    assert engine.combustor_exit_temperature_K == pytest.approx(1100.0)
    assert engine.turbine_isentropic_efficiency == pytest.approx(0.82)
    assert engine.mass_flow_air_kg_s == pytest.approx(0.15)
    expected = solve_brayton_cycle(FlightConditions(), engine)

    source = _source_frame(1)
    for name, value in row.items():
        source.loc[0, name] = value
    source.loc[0, "thrust_N"] = expected.thrust_N
    source.loc[0, "tsfc_kg_N_s"] = expected.tsfc_kg_N_s
    all_view, valid_view, _ = build_generator_views(
        source, schema, "test_v1", "source.csv"
    )
    _, valid_view, _ = assign_splits(
        all_view, valid_view, 42, 1.0, 0.0, 0.0
    )
    report, summary = run_parity_check(
        valid_view,
        schema,
        {
            "thrust_absolute_tolerance": 1e-12,
            "thrust_relative_tolerance": 1e-12,
            "tsfc_absolute_tolerance": 1e-14,
            "tsfc_relative_tolerance": 1e-12,
        },
    )
    assert summary["pass_count"] == 1
    assert bool(report.iloc[0]["parity_pass"]) is True


def test_wp1_small_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    source = generate_dataset(
        n_samples=80,
        output_dir=str(raw_dir),
        seed=19,
        verbose=False,
    )
    source_path = raw_dir / "dataset_80.csv"
    config = {
        "dataset_version": "test_generator_v1",
        "schema": str(SCHEMA_PATH),
        "source": {
            "csv": str(source_path),
            "source_id": "test/dataset_80.csv",
        },
        "output": {
            "directory": "unused",
            "split_csv": "unused.csv",
            "refuse_overwrite": True,
        },
        "split": {
            "seed": 42,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "parity": {
            "thrust_absolute_tolerance": 1e-10,
            "thrust_relative_tolerance": 1e-10,
            "tsfc_absolute_tolerance": 1e-12,
            "tsfc_relative_tolerance": 1e-10,
        },
    }
    config_path = tmp_path / "wp1.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    output_dir = run_wp1(
        config_path,
        repo_root=REPO_ROOT,
        output_dir_override=tmp_path / "generator_v1",
    )

    all_view = pd.read_csv(output_dir / "generator_dataset_v1_all.csv")
    valid_view = pd.read_csv(output_dir / "generator_dataset_v1_valid.csv")
    parity = pd.read_csv(output_dir / "parity_report.csv")
    assert len(all_view) == len(source)
    assert len(valid_view) == int(all_view["is_valid"].sum())
    assert parity["parity_pass"].all()
    assert (output_dir / "normalizer_v1.json").is_file()
    assert (output_dir / "generator_v1_split.csv").is_file()
    assert (output_dir / "dataset_card.md").is_file()
    assert (output_dir / "wp1_manifest.yaml").is_file()

    with pytest.raises(FileExistsError):
        run_wp1(
            config_path,
            repo_root=REPO_ROOT,
            output_dir_override=output_dir,
        )
