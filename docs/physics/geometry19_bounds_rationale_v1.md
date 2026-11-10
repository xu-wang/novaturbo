# Geometry19 Schema V1 — Bounds Rationale

## Status and scope

This document records the WP19-A design-space decision behind
`config/physics/geometry19_schema_v1.yaml`. The bounds freeze the first
Geometry-aware 19D implementation interface; they do not claim experimental
validity. Physics-domain limits remain the responsibility of the versioned
calibration pack and the forward evaluator constraints.

The bounds deliberately retain the established NovaTurbo micro-turbojet scale
around the reference design while replacing the old independently sampled
cycle states with physical geometry. The frozen inverse conditions remain
`[thrust_N, tsfc_kg_N_s, mass_kg]` and are not part of the 19D design vector.

## Frozen coordinate order and bounds

| # | Schema field | Bounds | Reference | Rationale |
|---:|---|---:|---:|---|
| 1 | `compressor_impeller_exit_diameter_m` | 60–150 mm | 80 mm | Preserves the former compressor diameter DOE range and the current 80 mm impeller. |
| 2 | `compressor_inducer_tip_diameter_m` | 35–100 mm | 50 mm | Contains the current 50 mm inducer while allowing scaling; the cross-constraint enforces `D_c1t < D_c2`. |
| 3 | `compressor_inducer_hub_tip_ratio` | 0.25–0.65 | 0.40 | Brackets the current 20/50 mm ratio without approaching a closed or vanishing eye. |
| 4 | `compressor_impeller_exit_width_m` | 3–12 mm | 5 mm | Covers the current small centrifugal-compressor width scale and supplies the missing outlet flow area. |
| 5 | `compressor_exit_blade_angle_deg` | 45–90 deg | 60 deg | Covers backward-swept through radial blades under the frozen tangential-angle convention. |
| 6 | `compressor_blade_count` | 8–20 | 12 | Reuses the existing integer DOE range and current baseline count. |
| 7 | `combustor_outer_diameter_m` | 80–160 mm | 110 mm | Reuses the established annular-combustor outer-diameter range. |
| 8 | `combustor_inner_diameter_m` | 40–90 mm | 60 mm | Reuses the former full-19D range and current baseline. |
| 9 | `combustor_length_m` | 50–150 mm | 80 mm | Reuses the established combustor length range. |
| 10 | `combustor_liner_thickness_m` | 0.8–2.0 mm | 1.2 mm | Reuses the DMLS liner range and current baseline thickness. |
| 11 | `combustor_injector_count` | 4–12 | 6 | Reuses the existing integer range and current six-injector design. |
| 12 | `turbine_tip_diameter_m` | 65–160 mm | 85 mm | Adds the missing turbine scale around the current 85 mm stage and covers the compressor/combustor envelope range. |
| 13 | `turbine_hub_tip_ratio` | 0.60–0.85 | 0.70 | Reuses the established turbine annulus range. |
| 14 | `turbine_blade_count` | 11–23 | 17 | Reuses the existing integer range and current baseline count. |
| 15 | `turbine_inlet_total_temperature_K` | 800–1200 K | 1100 K | Retains the current design-point TIT domain and configured material ceiling. |
| 16 | `nozzle_throat_diameter_m` | 30–80 mm | 55 mm | Reuses the former nozzle exit range; for the frozen convergent nozzle this is the throat and exit diameter. |
| 17 | `shaft_speed_rpm` | 60,000–150,000 rpm | 100,000 rpm | Reuses the existing micro-turbojet speed domain; tip-speed constraints reject incompatible diameter/speed pairs. |
| 18 | `lattice_cell_size_m` | 4–14 mm | 8 mm | Reuses the former TPMS cell-size DOE range. It affects manufacturing/structure constraints, not mass directly. |
| 19 | `lattice_relative_density` | 0.15–0.55 | 0.30 | Reuses the former lattice fill-fraction range and current reference value. |

The 2026-08-24 literature screening found common TPMS cell sizes around
`5-10 mm` (with published applications extending toward `6-12 mm`) and
relative densities around `0.10-0.50`. The existing `4-14 mm` and `0.15-0.55`
bounds are therefore retained as broad DOE envelopes rather than silently
changing the frozen schema. Machine-, material-, and topology-specific process
qualification may later justify a narrower schema version.

## Unit and type decisions

All lengths are stored in metres. User interfaces may display millimetres but
must convert before schema validation. TIT is stored in kelvin. Blade and
injector counts are integer-valued and use balanced discrete sampling rather
than continuous LHS rounding. Shaft speed remains in rpm because that is the
frozen V2.1 interface; the geometry layer must convert it with
`omega = 2*pi*N/60`.

## Independent bounds versus coupled feasibility

Independent bounds intentionally form a broad rectangular sampling envelope.
They do not imply that every Cartesian combination is a valid engine. The
schema distinguishes:

- `pre_solver_hard`: algebraically invalid geometry, rejected before physics;
- `post_geometry_feasibility`: a calculable design that violates component,
  assembly, structural, or manufacturing limits;
- solver failure: a valid geometry for which no unique closed operating point
  exists.

In particular, the upper diameter and RPM bounds can exceed the permitted tip
speed when combined. Those samples are retained as infeasible rather than
silently shrinking the independent bounds.

## Derived diffuser decision

The impeller exit area is not compared directly with the combustor annulus.
GeometryState must derive an unvaned/aggregate diffuser without adding a 20th
design variable:

\[
A_{d,out}=\Pi_{A,d}A_{c2},\qquad L_d=k_{Ld}D_{c2}.
\]

The diffuser-to-combustor assembly condition is evaluated using
`A_d_out/A_b_in`. The diffuser pressure loss remains included in the calibrated
compressor total-to-total efficiency closure for V2.1.

## Provenance and revision rule

Established bounds come from `config/constraints.yaml`,
`config/default_engine.yaml`, and the former full-19D bounds in
`src/ai/dataset.py`. New geometry bounds are bootstrap engineering ranges around
the current parametric compressor and turbine defaults. Calibration validity
ranges must be narrower where evidence requires it.

Any coordinate reorder, unit change, type change, or bound change requires a
new schema version and new dataset/normalizer/checkpoint provenance. The
current schema intentionally sets `formal_dataset_generation_ready: false`
until the calibration candidate, single-point conservation tests, and sanity
DOE gates pass.
