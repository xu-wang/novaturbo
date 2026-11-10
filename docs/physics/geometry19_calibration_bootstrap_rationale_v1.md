# Geometry19 Bootstrap Calibration Candidate V1

## Purpose

This pack makes the frozen V2.1 equations executable for implementation and
single-point validation. It is deliberately marked `bootstrap_candidate`.
It is not a final calibration and cannot authorize formal DOE generation.

Every model parameter records a value, unit, admissible calibration range,
source type, source reference, physical meaning, and intended calibration
target. Values tagged `PROJECT_BOOTSTRAP_ASSUMPTION` are hypotheses that must
be challenged by conservation tests, component-domain audits, and the sanity
DOE.

## Literature-screening revision 2026-08-24

An open-literature synthesis was screened after WP19-10. A trial narrowed the
compressor loading window to `0.45-0.75`, moved the radial-inflow turbine
design flow coefficient to `0.25`, lowered the turbine loading ceiling from
`1.8` to `1.15`, and raised the retained lattice skin minimum from `0.5 mm` to
`0.8 mm`. The identical-seed sanity DOE showed that these typical/optimum
literature regions are not interchangeable with the current model's calibrated
hard-domain definitions. The numerical trial was therefore rolled back: the
active bootstrap values remain `0.35-0.95`, `0.14`, `1.8`, and `0.5 mm`,
respectively. Inconel 718 room-temperature yield likewise remains at the
pre-review `1034 MPa` baseline rather than changing provenance and value in one
step.

The combustor pressure-loss ceiling remains `8%`, which is permissive but
inside the reported `2-10%` envelope. Literature and material-standard context
is retained as calibration evidence and definition warnings; it does not by
itself authorize replacement of the current project limits.

Values were deliberately not copied where definitions differ. In particular,
published centrifugal-compressor flow coefficients near `0.03-0.05` do not
use the same normalization as NovaTurbo's equivalent `V_m/U` coefficient.
Likewise, a rotor burst-speed margin of `1.2-1.25` is not the same quantity as
the current yield-utilization safety factor of `1.5`. These remain explicit
WP19-11 definition-audit items rather than silent numerical substitutions.

## Evidence hierarchy

1. Existing NovaTurbo configuration and parametric geometry provide reference
   dimensions, material records, TIT, RPM, efficiencies, and legacy limits.
2. Standard equations and the Stanitz slip baseline define equation form, not
   the project-specific loss coefficients.
3. Missing loss, mass-fill, assembly, thermal-proxy, and lattice coefficients
   receive explicit bootstrap assumptions rather than undocumented constants.
4. Experiment or high-fidelity evidence should replace bootstrap assumptions
   without changing the frozen 19D coordinate order.

## Corrected turbine mass flux

The pack freezes the reference conditions at 288.15 K and 101325 Pa and stores
the formal turbine capacity indicator as
`turbine_corrected_mass_flux_min/max`, in `kg/(s*m^2)`. It is not
dimensionless.

## Derived diffuser

The candidate uses

\[
A_{d,out}=2.5A_{c2},\qquad L_d=0.4D_{c2},
\]

with a target \(A_{d,out}/A_{b,in}=0.45\). These are assembly bootstrap
values; diffuser aerodynamic loss remains inside the whole-compressor
total-to-total efficiency closure.

## Thermodynamic consistency

The current NovaTurbo hot-gas constants are retained for regression continuity.
The relative mismatch between `cp_hot` and
\(\gamma_hR_g/(\gamma_h-1)\) is required by tests to remain below one
percent. The common sensible-enthalpy reference is 298.15 K.

The Reynolds-number closures require viscosity in addition to the caloric
properties. The cold side uses the standard Sutherland air reference
\(\mu_{ref}=1.716\times10^{-5}\,\mathrm{Pa\,s}\),
\(T_{ref}=273.15\,\mathrm{K}\), and \(S=110.4\,\mathrm{K}\). The hot-side
records are deliberately separate and tagged as bootstrap assumptions: they
currently reuse air-like values and must be replaced or validated for the
chosen combustion-product model before calibration freeze.

## Runtime ownership and provenance

The sample-wise environment contains only ambient temperature, ambient
pressure, and flight Mach number. Inlet total-pressure recovery
`inlet_total_pressure_recovery` is owned by this calibration pack rather than
the environment, so its provenance and calibrated range cannot vary silently
between samples.

Runtime configuration uses two SHA-256 identifiers. `source_file_hash` records
the exact YAML bytes, while `calibration_pack_hash` records the canonicalized
effective configuration after validated overrides. Any effective parameter or
solver-setting change therefore produces a different model provenance hash.

Parameter-level `notes` are optional. Their omission means that
`source_reference`, `physical_meaning`, and `calibration_target` completely
state the current qualification; `notes` are reserved for additional caveats.

## WP19-8 mass and constraint closures

The V2.1 mass model is evaluated analytically from Geometry19 and the `theta_m`
and `theta_s` records. Lattice cell size does not enter mass; relative density
enters the combustor and nozzle effective densities. The smallest qualified
lattice region is derived without a new design coordinate as
`min(combustor_hydraulic_diameter, nozzle_length)`.

The current pack contains literature-supported room-temperature yield
strengths but no validated
temperature-dependent allowable curves. WP19-8 therefore labels these values
as explicit bootstrap stress-screening proxies and independently enforces the
compressor, combustor, and turbine metal-temperature limits. Elevated-
temperature allowable curves remain a calibration-freeze requirement and must
replace the proxy before formal dataset generation is authorized.

## Release rule

The pack may advance from `bootstrap_candidate` to `validation_candidate`
only after `solve_design_point()` passes energy, shaft-power, flow, and nozzle
state tests. It may become `frozen` only after the small DOE establishes
acceptable convergence, feasibility, failure distribution, sensitivity, and
calibration-domain coverage. A frozen pack requires a content hash and a new
physics/dataset provenance record.
