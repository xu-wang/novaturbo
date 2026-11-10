#!/usr/bin/env python3
"""Run the complete WP19-23 target-to-visualization pipeline.

WP19_23_PIPELINE_SYNC_REQUIRED: every new WP19-23 runtime feature must be
added to this CLI and its orchestrator in the same change.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_pipeline_dispatch import (
    MODEL_CHOICES, MODEL_V2_1, MODEL_V3, run_wp19_23_pipeline,
)
from src.ai.geometry19_pipeline_v3 import Geometry19V3PendingError


def build_pipeline_context(*args, **kwargs):
    from src.ai.geometry19_pipeline_context import build_pipeline_context as build
    return build(*args, **kwargs)


def load_pipeline_profile_registry(*args, **kwargs):
    from src.ai.geometry19_pipeline_profiles import load_pipeline_profile_registry as load
    return load(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument("--model", choices=MODEL_CHOICES, default=MODEL_V2_1)
    model_parser.add_argument("--pipeline-profile", default=None)
    selected_model, _ = model_parser.parse_known_args(argv)

    # Only V2.1 obtains its options from a pipeline profile. The other model
    # entries own their complete protocols and must not construct V2.1 physics.
    registry = None
    pipeline_context = None
    protocol = None
    if selected_model.model == MODEL_V2_1:
        registry = load_pipeline_profile_registry(REPO_ROOT)
        profile_parser = argparse.ArgumentParser(add_help=False)
        profile_parser.add_argument(
            "--pipeline-profile", choices=registry.profile_names, default=None,
        )
        selected_profile, _ = profile_parser.parse_known_args(argv)
        pipeline_context = build_pipeline_context(
            REPO_ROOT,
            pipeline_profile=selected_profile.pipeline_profile or registry.default_profile,
            profile_registry=registry,
        )
        protocol = pipeline_context.protocol
    generator_variants = tuple(protocol["models"]["generator"]["variants"]) if protocol else None
    forward_model_tags = tuple(protocol["models"]["forward"]["variants"]) if protocol else None
    sampling_modes = tuple(protocol["generation"]["pipeline_sampling_modes"]) if protocol else None
    parser = argparse.ArgumentParser(
        description=(
            "WP19-23 Geometry19 target performance to pipeline result; "
            "candidate filtering uses unconditional nearest-integer projection and writes "
            "production raw/model provenance"
        )
    )
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=MODEL_V2_1,
        help="Select the V2.1 default, V2.2 local flow, or V3 22D development flow",
    )
    parser.add_argument("--thrust-N", type=float, default=None)
    parser.add_argument("--tsfc-kg-N-s", type=float, default=None)
    parser.add_argument("--mass-kg", type=float, default=None)
    parser.add_argument(
        "--pipeline-profile",
        choices=registry.profile_names if registry else None,
        default=None,
        help=("V2.1 profile; V2.2 accepts omitted or v2_1 for compatibility; "
              "V3 accepts v3_routed_development or v3_local_cfm_development; omitted uses fixtures"),
    )
    parser.add_argument(
        "--target-type",
        choices=("ID", "INTERPOLATION", "BOUNDARY_SPARSE", "OUT_OF_EMPIRICAL_SUPPORT"),
        default="ID",
    )
    parser.add_argument(
        "--expected-status",
        choices=("KNOWN_FEASIBLE", "UNKNOWN", "EXPECTED_INFEASIBLE"),
        default="UNKNOWN",
    )
    parser.add_argument(
        "--model-variant",
        choices=generator_variants,
        default=None,
        help="V2.1 generator variant; V2.2 and V3 reject an explicit value",
    )
    parser.add_argument(
        "--sampling",
        choices=sampling_modes,
        default=None,
        help=(
            "V2.1 profile sampling mode; V2.2 accepts only dflow; "
            "V3 rejects an explicit value"
        ),
    )
    parser.add_argument(
        "--forward-model-tag",
        choices=forward_model_tags,
        default=None,
        help="V2.1 forward model tag; V2.2 and V3 reject an explicit value",
    )
    parser.add_argument("--device", default=("cpu" if selected_model.model == MODEL_V3
                        and selected_model.pipeline_profile not in ("v3_routed_development", "v3_local_cfm_development", "v3_guided_cfm_development") else "cuda:0"))
    parser.add_argument("--refinement-device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--storage-root", type=Path, default=None)
    parser.add_argument(
        "--export-mesh",
        "--export-stl",
        dest="export_mesh",
        action="store_true",
        help="Export delivery meshes for V2.1/V2.2; V3 development does not support export",
    )
    restart_group = parser.add_mutually_exclusive_group()
    restart_group.add_argument("--resume", action="store_true")
    restart_group.add_argument(
        "--restart-from-stage",
        choices=("F",),
        help="V2.1 only: restart F after validating completed upstream stages",
    )
    args = parser.parse_args(argv)

    if args.model == MODEL_V3:
        for value, option in (
            (None if args.pipeline_profile in ("v3_routed_development", "v3_local_cfm_development", "v3_guided_cfm_development") else args.pipeline_profile, "--pipeline-profile"),
            (args.model_variant, "--model-variant"),
            (args.forward_model_tag, "--forward-model-tag"),
            (args.sampling, "--sampling"),
            (args.storage_root, "--storage-root"),
            (args.restart_from_stage, "--restart-from-stage"),
        ):
            if value is not None:
                parser.error(f"{option} is not available for model V3 development")
        if args.export_mesh:
            parser.error("--export-mesh is not available for model V3 development")

    values = (args.thrust_N, args.tsfc_kg_N_s, args.mass_kg)
    any_values = any(value is not None for value in values)
    all_values = all(value is not None for value in values)
    if args.target is not None and any_values:
        parser.error("Use --target or the three performance values, not both")
    if args.target is None and not all_values:
        parser.error("Provide --target or all of --thrust-N, --tsfc-kg-N-s, --mass-kg")
    target_values = (
        {
            "thrust_N": args.thrust_N,
            "tsfc_kg_N_s": args.tsfc_kg_N_s,
            "mass_kg": args.mass_kg,
        }
        if all_values
        else None
    )
    pipeline_profile = args.pipeline_profile
    if args.model == MODEL_V2_1 and pipeline_profile is None:
        pipeline_profile = registry.default_profile
    try:
        summary = run_wp19_23_pipeline(
            REPO_ROOT,
            target_path=args.target,
            target_values=target_values,
            target_type=args.target_type,
            expected_status=args.expected_status,
            model_variant=args.model_variant,
            forward_model_tag=args.forward_model_tag,
            sampling_mode=args.sampling,
            device=args.device,
            refinement_device=args.refinement_device,
            output_root=args.output_root,
            storage_root=args.storage_root,
            export_mesh=args.export_mesh,
            resume=args.resume,
            restart_from_stage=args.restart_from_stage,
            pipeline_profile=pipeline_profile,
            pipeline_context=pipeline_context,
            model=args.model,
        )
    except Geometry19V3PendingError as exc:
        parser.error(str(exc))
    print(
        f"WP19-23 pipeline: {summary['status']}; "
        f"source={summary['terminal_source_stage']}; "
        f"delivery={summary['delivery_status']}; "
        f"design_ids={','.join(summary['delivered_design_ids'])}"
    )
    print(f"runtime={summary['paths']['runtime_root']}")
    print(f"manifest={summary['paths']['pipeline_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
