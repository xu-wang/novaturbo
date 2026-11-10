"""Operating-point result contract and V2.1 flow-root search machinery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from src.physics.geometry_state import (
    Geometry19,
    GeometryState,
    GeometryStateError,
    derive_geometry_state,
)
from src.physics.model_config import CalibrationPackError, ModelConfig


class FailureCode(str, Enum):
    INVALID_GEOMETRY_INPUT = "INVALID_GEOMETRY_INPUT"
    INVALID_ENVIRONMENT = "INVALID_ENVIRONMENT"
    INVALID_CALIBRATION = "INVALID_CALIBRATION"
    MASS_MODEL_INVALID = "MASS_MODEL_INVALID"
    FORMAL_CONSTRAINT_INVALID = "FORMAL_CONSTRAINT_INVALID"
    GEOMETRY_INVALID = "GEOMETRY_INVALID"
    COMPRESSOR_BLOCKAGE_INVALID = "COMPRESSOR_BLOCKAGE_INVALID"
    COMPRESSOR_SLIP_INVALID = "COMPRESSOR_SLIP_INVALID"
    COMPRESSOR_NONPOSITIVE_WORK = "COMPRESSOR_NONPOSITIVE_WORK"
    COMPRESSOR_EFFICIENCY_INVALID = "COMPRESSOR_EFFICIENCY_INVALID"
    COMBUSTOR_PRESSURE_INVALID = "COMBUSTOR_PRESSURE_INVALID"
    COMBUSTOR_ENERGY_INVALID = "COMBUSTOR_ENERGY_INVALID"
    TURBINE_POWER_INSUFFICIENT = "TURBINE_POWER_INSUFFICIENT"
    TURBINE_EFFICIENCY_INVALID = "TURBINE_EFFICIENCY_INVALID"
    TURBINE_PRESSURE_INVALID = "TURBINE_PRESSURE_INVALID"
    NOZZLE_STATE_INVALID = "NOZZLE_STATE_INVALID"
    NO_FLOW_MATCH = "NO_FLOW_MATCH"
    MULTIPLE_OPERATING_POINTS = "MULTIPLE_OPERATING_POINTS"
    FLOW_RESIDUAL_DISCONTINUITY = "FLOW_RESIDUAL_DISCONTINUITY"
    NONFINITE_STATE = "NONFINITE_STATE"
    ROOT_SOLVER_FAILED = "ROOT_SOLVER_FAILED"


class OperatingPointSelectionPolicy(str, Enum):
    """Explicit policy for resolving otherwise valid flow-match roots."""

    UNIQUE_ONLY = "unique_only"
    REFERENCE_CONTINUATION_HIGH_FLOW_V1 = (
        "reference_continuation_high_flow_v1"
    )


@dataclass(frozen=True)
class Environment:
    temperature_K: float = 288.15
    pressure_Pa: float = 101325.0
    mach: float = 0.0


@dataclass(frozen=True)
class StationState:
    total_temperature_K: float
    total_pressure_Pa: float


@dataclass(frozen=True)
class NozzleExitState:
    static_temperature_K: float
    static_pressure_Pa: float
    mach: float
    velocity_m_s: float


@dataclass(frozen=True)
class ConservationResiduals:
    combustor_energy_W: float = math.nan
    shaft_power_W: float = math.nan
    mass_flow_kg_s: float = math.nan
    normalized_flow: float = math.nan
    nozzle_velocity_m_s: float = math.nan


@dataclass(frozen=True)
class SolverMetadata:
    converged: bool
    flow_residual: float = math.nan
    iterations: int = 0
    evaluations: int = 0
    failure_code: Optional[FailureCode] = None
    message: str = ""


@dataclass(frozen=True)
class EngineResult:
    """Typed counterpart of the EngineResult contract frozen in V2.1."""

    design: Optional[Geometry19]
    environment: Optional[Environment]
    operating_point: Mapping[str, float] = field(default_factory=dict)
    stations: Mapping[str, StationState | NozzleExitState] = field(
        default_factory=dict
    )
    components: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    performance: Mapping[str, float] = field(default_factory=dict)
    constraints: Mapping[str, float] = field(default_factory=dict)
    mass_breakdown_kg: Mapping[str, float] = field(default_factory=dict)
    mass_volume_breakdown_m3: Mapping[str, float] = field(default_factory=dict)
    formal_diagnostics: Mapping[str, float | str | bool] = field(
        default_factory=dict
    )
    feasible: bool = False
    core_feasible: bool = False
    thermodynamic_feasible: bool = False
    thermodynamic_constraints: Mapping[str, float] = field(default_factory=dict)
    assembly_constraints: Mapping[str, float] = field(default_factory=dict)
    structural_constraints: Mapping[str, float] = field(default_factory=dict)
    thermal_constraints: Mapping[str, float] = field(default_factory=dict)
    manufacturing_constraints: Mapping[str, float] = field(default_factory=dict)
    assembly_diagnostics: Mapping[str, float] = field(default_factory=dict)
    structural_diagnostics: Mapping[str, float] = field(default_factory=dict)
    assembly_diagnostics_pending: bool = True
    structural_diagnostics_pending: bool = True
    feasibility_complete: bool = False
    feasibility_scope: str = "geometry19_wp19_8_not_evaluated"
    conservation: ConservationResiduals = field(
        default_factory=ConservationResiduals
    )
    solver: SolverMetadata = field(
        default_factory=lambda: SolverMetadata(converged=False)
    )
    physics_model_version: str = "geometry19_closed_v2_1"
    calibration_pack_hash: str = ""
    calibration_status: str = "unknown"
    formal_dataset_generation_ready: bool = False


@dataclass(frozen=True)
class TrialEvaluation:
    """One mass-flow trial, either a complete valid state or hard failure."""

    mass_flow_air_kg_s: float
    valid: bool
    flow_residual: Optional[float] = None
    failure_code: Optional[FailureCode] = None
    payload: Any = None

    @classmethod
    def valid_state(
        cls, mass_flow_air_kg_s: float, flow_residual: float, payload: Any = None
    ) -> "TrialEvaluation":
        if not math.isfinite(flow_residual):
            return cls.hard_failure(
                mass_flow_air_kg_s, FailureCode.NONFINITE_STATE, payload
            )
        return cls(
            mass_flow_air_kg_s=mass_flow_air_kg_s,
            valid=True,
            flow_residual=float(flow_residual),
            payload=payload,
        )

    @classmethod
    def hard_failure(
        cls,
        mass_flow_air_kg_s: float,
        failure_code: FailureCode,
        payload: Any = None,
    ) -> "TrialEvaluation":
        return cls(
            mass_flow_air_kg_s=mass_flow_air_kg_s,
            valid=False,
            failure_code=failure_code,
            payload=payload,
        )


@dataclass(frozen=True)
class RootSearchConfig:
    linear_scan_points: int = 64
    logarithmic_scan_points: int = 64
    residual_tolerance: float = 1.0e-5
    mass_flow_residual_epsilon_kg_s: float = 1.0e-12
    root_relative_tolerance: float = 1.0e-6
    tangency_refine_threshold: float = 1.0e-3

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "RootSearchConfig":
        raw = config.solver_candidate
        return cls(
            linear_scan_points=int(raw["linear_scan_points"]),
            logarithmic_scan_points=int(raw["logarithmic_scan_points"]),
            residual_tolerance=float(raw["flow_residual_tolerance"]),
            mass_flow_residual_epsilon_kg_s=float(
                raw["mass_flow_residual_epsilon_kg_s"]
            ),
            root_relative_tolerance=float(raw["root_relative_tolerance"]),
            tangency_refine_threshold=float(raw["tangency_refine_threshold"]),
        )


@dataclass(frozen=True)
class FlowRoot:
    mass_flow_air_kg_s: float
    flow_residual: float
    trial: TrialEvaluation


@dataclass(frozen=True)
class FlowMatchResult:
    converged: bool
    roots: tuple[FlowRoot, ...]
    failure_code: Optional[FailureCode]
    evaluations: int
    iterations: int


TrialEvaluator = Callable[[float], TrialEvaluation]


@dataclass(frozen=True)
class DesignPointTrialState:
    """Complete component chain evaluated at one air mass-flow trial."""

    geometry: GeometryState
    inlet: StationState
    compressor: Any
    combustor: Any
    turbine: Any
    nozzle: Any
    flight_velocity_m_s: float


def find_flow_roots(
    evaluator: TrialEvaluator,
    mass_flow_min_kg_s: float,
    mass_flow_max_kg_s: float,
    search: RootSearchConfig,
) -> FlowMatchResult:
    """Find all V2.1 flow matches without bridging hard-failure domains."""
    if not (
        math.isfinite(mass_flow_min_kg_s)
        and math.isfinite(mass_flow_max_kg_s)
        and 0.0 < mass_flow_min_kg_s < mass_flow_max_kg_s
    ):
        return FlowMatchResult(
            converged=False,
            roots=(),
            failure_code=FailureCode.ROOT_SOLVER_FAILED,
            evaluations=0,
            iterations=0,
        )
    search_values = (
        search.residual_tolerance,
        search.mass_flow_residual_epsilon_kg_s,
        search.root_relative_tolerance,
        search.tangency_refine_threshold,
    )
    if (
        type(search.linear_scan_points) is not int
        or type(search.logarithmic_scan_points) is not int
        or search.linear_scan_points < 3
        or search.linear_scan_points > 4096
        or search.logarithmic_scan_points < 3
        or search.logarithmic_scan_points > 4096
        or not all(
            isinstance(value, Real) and not isinstance(value, bool)
            for value in search_values
        )
        or not all(math.isfinite(value) for value in search_values)
        or not 1.0e-10 <= search.residual_tolerance <= 1.0e-3
        or not (
            1.0e-16
            <= search.mass_flow_residual_epsilon_kg_s
            <= 1.0e-6
        )
        or not 1.0e-12 <= search.root_relative_tolerance <= 1.0e-2
        or not 1.0e-10 <= search.tangency_refine_threshold <= 1.0e-1
        or search.residual_tolerance > search.tangency_refine_threshold
    ):
        return FlowMatchResult(
            converged=False,
            roots=(),
            failure_code=FailureCode.ROOT_SOLVER_FAILED,
            evaluations=0,
            iterations=0,
        )

    scan_points = np.unique(
        np.concatenate(
            (
                np.linspace(
                    mass_flow_min_kg_s,
                    mass_flow_max_kg_s,
                    search.linear_scan_points,
                ),
                np.geomspace(
                    mass_flow_min_kg_s,
                    mass_flow_max_kg_s,
                    search.logarithmic_scan_points,
                ),
            )
        )
    )
    cache: dict[float, TrialEvaluation] = {}

    def evaluate(value: float) -> TrialEvaluation:
        key = float(value)
        if key not in cache:
            trial = evaluator(key)
            if not isinstance(trial, TrialEvaluation):
                raise TypeError("Flow trial evaluator must return TrialEvaluation")
            cache[key] = trial
        return cache[key]

    try:
        scan_trials = [evaluate(float(value)) for value in scan_points]
    except (ArithmeticError, TypeError, ValueError):
        return FlowMatchResult(
            converged=False,
            roots=(),
            failure_code=FailureCode.ROOT_SOLVER_FAILED,
            evaluations=len(cache),
            iterations=0,
        )
    candidates: list[tuple[float, int]] = []
    total_iterations = 0
    rejected_discontinuous_brackets = 0

    for trial in scan_trials:
        if (
            trial.valid
            and trial.flow_residual is not None
            and abs(trial.flow_residual) <= search.residual_tolerance
        ):
            candidates.append((trial.mass_flow_air_kg_s, 0))

    def scalar_residual(value: float) -> float:
        trial = evaluate(value)
        if not trial.valid or trial.flow_residual is None:
            raise ValueError("Hard failure encountered inside a candidate bracket")
        return trial.flow_residual

    for left, right in zip(scan_trials[:-1], scan_trials[1:]):
        if not left.valid or not right.valid:
            continue
        assert left.flow_residual is not None
        assert right.flow_residual is not None
        if left.flow_residual * right.flow_residual >= 0.0:
            continue
        try:
            root, result = brentq(
                scalar_residual,
                left.mass_flow_air_kg_s,
                right.mass_flow_air_kg_s,
                xtol=np.finfo(float).eps,
                rtol=max(search.root_relative_tolerance, 4.0 * np.finfo(float).eps),
                full_output=True,
                disp=False,
            )
        except (ValueError, RuntimeError):
            continue
        total_iterations += int(result.iterations)
        # A bracketing method can report convergence across a sign-changing
        # discontinuity.  Re-evaluate and admit only a finite residual that
        # satisfies the same physical closure gate as scan/tangency roots.
        try:
            root_trial = evaluate(float(root))
        except (ArithmeticError, TypeError, ValueError):
            continue
        if (
            root_trial.valid
            and root_trial.flow_residual is not None
            and math.isfinite(root_trial.flow_residual)
            and abs(root_trial.flow_residual) <= search.residual_tolerance
        ):
            candidates.append((float(root), int(result.iterations)))
        else:
            rejected_discontinuous_brackets += 1

    for index in range(1, len(scan_trials) - 1):
        left, middle, right = scan_trials[index - 1 : index + 2]
        if not (left.valid and middle.valid and right.valid):
            continue
        assert left.flow_residual is not None
        assert middle.flow_residual is not None
        assert right.flow_residual is not None
        middle_abs = abs(middle.flow_residual)
        if not (
            middle_abs <= abs(left.flow_residual)
            and middle_abs <= abs(right.flow_residual)
            and middle_abs < search.tangency_refine_threshold
        ):
            continue

        def absolute_residual(value: float) -> float:
            trial = evaluate(value)
            if not trial.valid or trial.flow_residual is None:
                return math.inf
            return abs(trial.flow_residual)

        try:
            refined = minimize_scalar(
                absolute_residual,
                bounds=(left.mass_flow_air_kg_s, right.mass_flow_air_kg_s),
                method="bounded",
                options={
                    "xatol": search.root_relative_tolerance
                    * middle.mass_flow_air_kg_s
                },
            )
        except (ArithmeticError, TypeError, ValueError):
            continue
        total_iterations += int(refined.nfev)
        if refined.success and refined.fun <= search.residual_tolerance:
            candidates.append((float(refined.x), int(refined.nfev)))

    candidates.sort(key=lambda item: item[0])
    unique_candidates: list[float] = []
    for value, _ in candidates:
        if not unique_candidates:
            unique_candidates.append(value)
            continue
        separation = abs(value - unique_candidates[-1])
        scale = max(abs(value), abs(unique_candidates[-1]), 1.0e-12)
        if separation > 5.0 * search.root_relative_tolerance * scale:
            unique_candidates.append(value)

    roots = []
    for value in unique_candidates:
        trial = evaluate(value)
        if (
            trial.valid
            and trial.flow_residual is not None
            and math.isfinite(trial.flow_residual)
            and abs(trial.flow_residual) <= search.residual_tolerance
        ):
            roots.append(
                FlowRoot(
                    mass_flow_air_kg_s=value,
                    flow_residual=trial.flow_residual,
                    trial=trial,
                )
            )

    if not roots and rejected_discontinuous_brackets:
        failure_code = FailureCode.FLOW_RESIDUAL_DISCONTINUITY
    elif not roots:
        failure_code = FailureCode.NO_FLOW_MATCH
    elif len(roots) > 1:
        failure_code = FailureCode.MULTIPLE_OPERATING_POINTS
    else:
        failure_code = None
    return FlowMatchResult(
        converged=len(roots) == 1,
        roots=tuple(roots),
        failure_code=failure_code,
        evaluations=len(cache),
        iterations=total_iterations,
    )


def _coerce_geometry19(
    x19: Geometry19 | Mapping[str, float | int] | Sequence[float | int],
) -> Geometry19:
    if isinstance(x19, Geometry19):
        return x19
    if isinstance(x19, Mapping):
        return Geometry19.from_mapping(x19)
    if isinstance(x19, Sequence) and not isinstance(x19, (str, bytes)):
        return Geometry19.from_sequence(x19)
    raise GeometryStateError(
        FailureCode.GEOMETRY_INVALID.value,
        "x19 must be Geometry19, a canonical mapping, or a 19-value sequence",
    )


def _failure_result(
    design: Optional[Geometry19],
    environment: Optional[Environment],
    config: Optional[ModelConfig],
    failure_code: FailureCode,
    message: str,
    *,
    evaluations: int = 0,
    iterations: int = 0,
    operating_point: Optional[Mapping[str, float]] = None,
) -> EngineResult:
    return EngineResult(
        design=design,
        environment=environment,
        operating_point=dict(operating_point or {}),
        feasible=False,
        core_feasible=False,
        feasibility_complete=False,
        solver=SolverMetadata(
            converged=False,
            evaluations=evaluations,
            iterations=iterations,
            failure_code=failure_code,
            message=message,
        ),
        physics_model_version=(
            config.physics_spec_version if config is not None else "geometry19_unknown"
        ),
        calibration_pack_hash=(
            config.calibration_pack_hash if config is not None else ""
        ),
        calibration_status=(config.status if config is not None else "unknown"),
        formal_dataset_generation_ready=(
            config.formal_dataset_generation_ready
            if config is not None
            else False
        ),
    )


def _inlet_state_and_capacity(
    geometry: GeometryState,
    environment: Environment,
    config: ModelConfig,
) -> tuple[StationState, float, float]:
    values = (
        environment.temperature_K,
        environment.pressure_Pa,
        environment.mach,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Environment contains non-finite values")
    if (
        environment.temperature_K <= 0.0
        or environment.pressure_Pa <= 0.0
        or environment.mach < 0.0
    ):
        raise ValueError(
            "Environment temperature and pressure must be positive and Mach nonnegative"
        )

    gamma = float(config.require("gamma_cold"))
    gas_constant = float(config.require("gas_constant_cold_J_kg_K"))
    pressure_recovery = float(config.require("inlet_total_pressure_recovery"))
    discharge_coefficient = float(config.require("inlet_discharge_coefficient"))
    if gamma <= 1.0 or gas_constant <= 0.0:
        raise ValueError("Cold-side inlet gas properties are invalid")
    if not 0.0 < pressure_recovery <= 1.0:
        raise ValueError("Inlet total-pressure recovery must be in (0, 1]")
    if not 0.0 < discharge_coefficient <= 1.0:
        raise ValueError("Inlet discharge coefficient must be in (0, 1]")

    ram_factor = 1.0 + 0.5 * (gamma - 1.0) * environment.mach**2
    total_temperature_0 = environment.temperature_K * ram_factor
    total_pressure_0 = environment.pressure_Pa * ram_factor ** (
        gamma / (gamma - 1.0)
    )
    inlet = StationState(
        total_temperature_K=total_temperature_0,
        total_pressure_Pa=pressure_recovery * total_pressure_0,
    )
    flight_velocity = environment.mach * math.sqrt(
        gamma * gas_constant * environment.temperature_K
    )
    choked_factor = math.sqrt(gamma / gas_constant) * (
        2.0 / (gamma + 1.0)
    ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    maximum_air_flow = (
        discharge_coefficient
        * geometry.compressor_inlet_effective_area_m2
        * inlet.total_pressure_Pa
        / math.sqrt(inlet.total_temperature_K)
        * choked_factor
    )
    if not math.isfinite(maximum_air_flow) or maximum_air_flow <= 0.0:
        raise ValueError("Inlet choked-capacity bracket is invalid")
    return inlet, flight_velocity, maximum_air_flow


def _component_failure_code(error: Exception) -> FailureCode:
    raw_code = getattr(error, "failure_code", FailureCode.ROOT_SOLVER_FAILED)
    if isinstance(raw_code, FailureCode):
        return raw_code
    try:
        return FailureCode(str(raw_code))
    except ValueError:
        return FailureCode.ROOT_SOLVER_FAILED


def solve_design_point(
    x19: Geometry19 | Mapping[str, float | int] | Sequence[float | int],
    environment: Environment,
    model_config: ModelConfig,
    *,
    operating_point_policy: OperatingPointSelectionPolicy | str = (
        OperatingPointSelectionPolicy.UNIQUE_ONLY
    ),
    inlet_solver: Callable[..., Any] | None = None,
    compressor_solver: Callable[..., Any] | None = None,
    combustor_solver: Callable[..., Any] | None = None,
    turbine_solver: Callable[..., Any] | None = None,
    nozzle_solver: Callable[..., Any] | None = None,
) -> EngineResult:
    """Solve one Geometry19 operating point by matching gas flow to nozzle capacity.

    WP19-7 supplies the matched thermodynamic core. WP19-8 then evaluates the
    analytic mass closure and all assembly, structural, thermal, and
    manufacturing constraints before returning formal feasibility.
    """
    if not isinstance(model_config, ModelConfig):
        return _failure_result(
            None,
            environment if isinstance(environment, Environment) else None,
            None,
            FailureCode.INVALID_CALIBRATION,
            "model_config must be a validated ModelConfig",
        )
    try:
        selection_policy = OperatingPointSelectionPolicy(operating_point_policy)
    except (TypeError, ValueError):
        return _failure_result(
            None,
            environment if isinstance(environment, Environment) else None,
            model_config,
            FailureCode.INVALID_CALIBRATION,
            f"Unsupported operating-point selection policy: {operating_point_policy!r}",
        )
    try:
        design = _coerce_geometry19(x19)
    except (GeometryStateError, TypeError, ValueError) as exc:
        return _failure_result(
            None,
            environment if isinstance(environment, Environment) else None,
            model_config,
            FailureCode.INVALID_GEOMETRY_INPUT,
            str(exc),
        )
    if not isinstance(environment, Environment):
        return _failure_result(
            design,
            None,
            model_config,
            FailureCode.INVALID_ENVIRONMENT,
            "environment must be an Environment",
        )
    environment_values = (
        environment.temperature_K,
        environment.pressure_Pa,
        environment.mach,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in environment_values
    ):
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.INVALID_ENVIRONMENT,
            "Environment values must be real numbers",
        )
    if not all(math.isfinite(float(value)) for value in environment_values):
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.NONFINITE_STATE,
            "Environment contains non-finite values",
        )
    if (
        environment.temperature_K <= 0.0
        or environment.pressure_Pa <= 0.0
        or environment.mach < 0.0
    ):
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.INVALID_ENVIRONMENT,
            "Environment temperature and pressure must be positive and Mach "
            "nonnegative",
        )
    try:
        geometry = derive_geometry_state(design, model_config)
    except GeometryStateError as exc:
        return _failure_result(
            design,
            environment,
            model_config,
            _component_failure_code(exc),
            str(exc),
        )
    except CalibrationPackError as exc:
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.INVALID_CALIBRATION,
            str(exc),
        )
    except ArithmeticError as exc:
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.NONFINITE_STATE,
            str(exc),
        )

    try:
        resolved_inlet_solver = inlet_solver or _inlet_state_and_capacity
        inlet, flight_velocity, inlet_capacity = resolved_inlet_solver(
            geometry, environment, model_config
        )
        lower_fraction = float(
            model_config.solver_candidate["mass_flow_lower_fraction"]
        )
        upper_fraction = float(
            model_config.solver_candidate["mass_flow_upper_fraction"]
        )
        if not 0.0 < lower_fraction < upper_fraction <= 1.0:
            raise ValueError("Mass-flow bracket fractions are invalid")
        mass_flow_min = lower_fraction * inlet_capacity
        mass_flow_max = upper_fraction * inlet_capacity
        search = RootSearchConfig.from_model_config(model_config)
    except CalibrationPackError as exc:
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.INVALID_CALIBRATION,
            str(exc),
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.NONFINITE_STATE,
            str(exc),
        )

    from src.physics.combustor_loworder import (
        CombustorModelError,
        solve_combustor_loworder,
    )
    from src.physics.compressor_meanline import (
        CompressorModelError,
        solve_compressor_meanline,
    )
    from src.physics.nozzle import NozzleModelError, solve_convergent_nozzle
    from src.physics.turbine_meanline import (
        TurbineModelError,
        solve_turbine_meanline,
    )

    component_errors = (
        CompressorModelError,
        CombustorModelError,
        TurbineModelError,
        NozzleModelError,
    )
    resolved_compressor_solver = compressor_solver or solve_compressor_meanline
    resolved_combustor_solver = combustor_solver or solve_combustor_loworder
    resolved_turbine_solver = turbine_solver or solve_turbine_meanline
    resolved_nozzle_solver = nozzle_solver or solve_convergent_nozzle

    def evaluate_trial(mass_flow_air: float) -> TrialEvaluation:
        try:
            compressor = resolved_compressor_solver(
                geometry, mass_flow_air, inlet, model_config
            )
            combustor = resolved_combustor_solver(
                geometry, mass_flow_air, compressor.outlet, model_config
            )
            turbine = resolved_turbine_solver(
                geometry, combustor, compressor.shaft_power_W, model_config
            )
            nozzle = resolved_nozzle_solver(
                geometry, turbine, environment, model_config
            )
        except component_errors as exc:
            return TrialEvaluation.hard_failure(
                mass_flow_air,
                _component_failure_code(exc),
                {"message": str(exc)},
            )
        except CalibrationPackError as exc:
            return TrialEvaluation.hard_failure(
                mass_flow_air,
                FailureCode.INVALID_CALIBRATION,
                {"message": str(exc)},
            )
        except (ArithmeticError, ValueError) as exc:
            return TrialEvaluation.hard_failure(
                mass_flow_air,
                FailureCode.NONFINITE_STATE,
                {"message": str(exc)},
            )
        payload = DesignPointTrialState(
            geometry=geometry,
            inlet=inlet,
            compressor=compressor,
            combustor=combustor,
            turbine=turbine,
            nozzle=nozzle,
            flight_velocity_m_s=flight_velocity,
        )
        return TrialEvaluation.valid_state(
            mass_flow_air,
            nozzle.normalized_flow_residual,
            payload,
        )

    try:
        flow_match = find_flow_roots(
            evaluate_trial,
            mass_flow_min,
            mass_flow_max,
            search,
        )
    except (ArithmeticError, TypeError, ValueError) as exc:
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.ROOT_SOLVER_FAILED,
            str(exc),
        )
    bracket = {
        "mass_flow_search_min_kg_s": mass_flow_min,
        "mass_flow_search_max_kg_s": mass_flow_max,
        "inlet_choked_capacity_kg_s": inlet_capacity,
        "flow_root_count": len(flow_match.roots),
        "flow_root_mass_flow_air_kg_s": [
            root.mass_flow_air_kg_s for root in flow_match.roots
        ],
        "flow_root_normalized_residuals": [
            root.flow_residual for root in flow_match.roots
        ],
        "flow_root_selection_policy": selection_policy.value,
        "selected_flow_root_index": None,
        "selected_flow_root_mass_flow_air_kg_s": None,
    }
    selected_root_index: int | None = None
    if len(flow_match.roots) == 1:
        selected_root_index = 0
    elif (
        len(flow_match.roots) > 1
        and selection_policy
        is OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1
    ):
        selected_root_index = max(
            range(len(flow_match.roots)),
            key=lambda index: flow_match.roots[index].mass_flow_air_kg_s,
        )
    if selected_root_index is None:
        return _failure_result(
            design,
            environment,
            model_config,
            flow_match.failure_code or FailureCode.ROOT_SOLVER_FAILED,
            "No unique valid whole-engine flow match was found",
            evaluations=flow_match.evaluations,
            iterations=flow_match.iterations,
            operating_point=bracket,
        )

    root = flow_match.roots[selected_root_index]
    bracket.update(
        {
            "selected_flow_root_index": selected_root_index,
            "selected_flow_root_mass_flow_air_kg_s": root.mass_flow_air_kg_s,
        }
    )
    if (
        not math.isfinite(root.flow_residual)
        or abs(root.flow_residual) > search.residual_tolerance
    ):
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.FLOW_RESIDUAL_DISCONTINUITY,
            "Flow candidate failed final residual validation",
            evaluations=flow_match.evaluations,
            iterations=flow_match.iterations,
            operating_point=bracket,
        )
    trial = root.trial.payload
    if not isinstance(trial, DesignPointTrialState):
        return _failure_result(
            design,
            environment,
            model_config,
            FailureCode.ROOT_SOLVER_FAILED,
            "Flow root did not retain its component state",
            evaluations=flow_match.evaluations,
            iterations=flow_match.iterations,
            operating_point=bracket,
        )

    compressor = trial.compressor
    combustor = trial.combustor
    turbine = trial.turbine
    nozzle = trial.nozzle
    thrust = (
        combustor.mass_flow_gas_kg_s * nozzle.exit.velocity_m_s
        - root.mass_flow_air_kg_s * flight_velocity
        + nozzle.pressure_thrust_N
    )
    tsfc = (
        combustor.mass_flow_fuel_kg_s / thrust
        if math.isfinite(thrust) and thrust > 0.0
        else math.inf
    )

    from src.physics.formal_constraints_v2 import (
        FormalConstraintError,
        evaluate_formal_constraints_v2,
    )
    from src.physics.mass_model_v2 import MassModelError, solve_mass_model_v2

    try:
        mass = solve_mass_model_v2(geometry, model_config)
        formal = evaluate_formal_constraints_v2(
            geometry,
            compressor.outlet,
            combustor,
            mass,
            model_config,
        )
    except (MassModelError, FormalConstraintError) as exc:
        return _failure_result(
            design,
            environment,
            model_config,
            _component_failure_code(exc),
            str(exc),
            evaluations=flow_match.evaluations,
            iterations=flow_match.iterations,
            operating_point={
                **bracket,
                "mass_flow_air_kg_s": root.mass_flow_air_kg_s,
                "mass_flow_fuel_kg_s": combustor.mass_flow_fuel_kg_s,
                "mass_flow_gas_kg_s": combustor.mass_flow_gas_kg_s,
            },
        )

    compressor_constraints = dict(compressor.calibration_domain_residuals)
    turbine_constraints = dict(turbine.constraint_residuals)
    structural_diagnostics = {
        "compressor_tip_speed_limit": compressor_constraints.pop(
            "compressor_tip_speed_limit"
        ),
        "turbine_tip_speed_limit": turbine_constraints.pop(
            "turbine_tip_speed_limit"
        ),
        **formal.structural_residuals,
    }
    inlet_constraints = dict(getattr(inlet, "constraint_residuals", {}))
    thermodynamic_constraints = {
        **inlet_constraints,
        **compressor_constraints,
        **combustor.constraint_residuals,
        **turbine_constraints,
        **nozzle.state_constraint_residuals,
        "whole_engine_flow_match": abs(nozzle.normalized_flow_residual)
        - search.residual_tolerance,
        "positive_thrust": -thrust,
        "positive_tsfc": -tsfc,
    }
    required_outputs_valid = (
        math.isfinite(thrust)
        and thrust > 0.0
        and math.isfinite(tsfc)
        and tsfc > 0.0
        and math.isfinite(mass.total_mass_kg)
        and mass.total_mass_kg > 0.0
    )
    thermodynamic_feasible = required_outputs_valid and all(
        math.isfinite(value) and value <= 0.0
        for value in thermodynamic_constraints.values()
    )
    all_constraints = {
        **thermodynamic_constraints,
        **geometry.assembly_residuals,
        **structural_diagnostics,
        **formal.thermal_residuals,
        **formal.manufacturing_residuals,
    }
    feasible = required_outputs_valid and all(
        math.isfinite(value) and value <= 0.0
        for value in all_constraints.values()
    )

    return EngineResult(
        design=design,
        environment=environment,
        operating_point={
            **bracket,
            "mass_flow_air_kg_s": root.mass_flow_air_kg_s,
            "mass_flow_fuel_kg_s": combustor.mass_flow_fuel_kg_s,
            "mass_flow_gas_kg_s": combustor.mass_flow_gas_kg_s,
            "fuel_air_ratio": combustor.fuel_air_ratio,
            "air_fuel_ratio": combustor.air_fuel_ratio,
        },
        stations={
            "t2": inlet,
            "t3": compressor.outlet,
            "t4": combustor.outlet,
            "t5": turbine.outlet,
            "nozzle_total": nozzle.nozzle_total,
            "exit": nozzle.exit,
        },
        components={
            **(
                {
                    "inlet": {
                        "map_id": inlet.map_id,
                        "map_version": inlet.map_version,
                        "map_source_hash": inlet.map_source_hash,
                        "map_in_domain": inlet.map_in_domain,
                        "map_extrapolation_distance": (
                            inlet.map_extrapolation_distance
                        ),
                        "map_uncertainty": dict(inlet.map_uncertainty),
                        "total_pressure_recovery": inlet.total_pressure_recovery,
                        "discharge_coefficient": inlet.discharge_coefficient,
                        "capture_area_ratio": inlet.capture_area_ratio,
                        "minimum_capture_margin": inlet.minimum_capture_margin,
                        "maximum_capture_margin": inlet.maximum_capture_margin,
                        "freestream_total_temperature_K": (
                            inlet.freestream_total_temperature_K
                        ),
                        "freestream_total_pressure_Pa": (
                            inlet.freestream_total_pressure_Pa
                        ),
                        "total_enthalpy_residual_J_kg": (
                            inlet.total_enthalpy_residual_J_kg
                        ),
                        "thermo_model_id": inlet.thermo_model_id,
                    }
                }
                if hasattr(inlet, "map_id")
                else {}
            ),
            "compressor": {
                "pressure_ratio": compressor.pressure_ratio,
                "total_to_total_efficiency": compressor.total_to_total_efficiency,
                "flow_coefficient_ref": compressor.flow_coefficient_ref,
                "loading_coefficient": compressor.loading_coefficient,
                "slip_factor": compressor.slip_factor,
                "shaft_power_W": compressor.shaft_power_W,
                **(
                    {
                        "map_id": compressor.map_id,
                        "map_version": compressor.map_version,
                        "map_source_hash": compressor.map_source_hash,
                        "map_in_domain": compressor.map_in_domain,
                        "map_extrapolation_distance": (
                            compressor.map_extrapolation_distance
                        ),
                        "map_uncertainty": dict(compressor.map_uncertainty),
                        "surge_margin": compressor.surge_margin,
                        "choke_margin": compressor.choke_margin,
                        "corrected_speed_ratio": compressor.corrected_speed_ratio,
                        "flow_coefficient_ratio": (
                            compressor.flow_coefficient_ratio
                        ),
                        "work_coefficient": compressor.work_coefficient,
                        "euler_work_factor": compressor.euler_work_factor,
                        "euler_specific_work_J_kg": (
                            compressor.euler_specific_work_J_kg
                        ),
                        "thermo_model_id": compressor.thermo_model_id,
                    }
                    if hasattr(compressor, "map_id")
                    else {}
                ),
            },
            "combustor": {
                "pressure_loss_fraction": combustor.total_pressure_loss_fraction,
                "primary_residence_time_s": combustor.primary_zone_residence_time_s,
                "equivalence_ratio": combustor.equivalence_ratio,
                **(
                    {
                        "combustion_efficiency": combustor.combustion_efficiency,
                        "mass_flux_ratio": combustor.mass_flux_ratio,
                        "map_id": combustor.map_id,
                        "map_version": combustor.map_version,
                        "map_source_hash": combustor.map_source_hash,
                        "map_in_domain": combustor.map_in_domain,
                        "map_extrapolation_distance": (
                            combustor.map_extrapolation_distance
                        ),
                        "map_uncertainty": dict(combustor.map_uncertainty),
                        "lean_stability_margin": (
                            combustor.lean_stability_margin
                        ),
                        "rich_stability_margin": (
                            combustor.rich_stability_margin
                        ),
                        "fuel_solution_iterations": (
                            combustor.fuel_solution_iterations
                        ),
                        "thermo_model_id": combustor.thermo_model_id,
                    }
                    if hasattr(combustor, "map_id")
                    else {}
                ),
            },
            "turbine": {
                "total_to_total_efficiency": turbine.total_to_total_efficiency,
                "flow_coefficient": turbine.flow_coefficient,
                "loading_coefficient": turbine.loading_coefficient,
                "mean_radius_mach": turbine.mean_radius_mach,
                "solidity": turbine.solidity,
                "corrected_mass_flux_kg_s_m2": turbine.corrected_mass_flux_kg_s_m2,
                **(
                    {
                        "map_id": turbine.map_id,
                        "map_version": turbine.map_version,
                        "map_source_hash": turbine.map_source_hash,
                        "map_in_domain": turbine.map_in_domain,
                        "map_extrapolation_distance": (
                            turbine.map_extrapolation_distance
                        ),
                        "map_uncertainty": dict(turbine.map_uncertainty),
                        "minimum_capacity_margin": (
                            turbine.minimum_capacity_margin
                        ),
                        "maximum_capacity_margin": (
                            turbine.maximum_capacity_margin
                        ),
                        "corrected_speed_ratio": turbine.corrected_speed_ratio,
                        "corrected_mass_flux_ratio": (
                            turbine.corrected_mass_flux_ratio
                        ),
                        **(
                            {
                                "thermo_model_id": turbine.thermo_model_id,
                                "outlet_enthalpy_residual_J_kg": (
                                    turbine.outlet_enthalpy_residual_J_kg
                                ),
                                "isentropic_efficiency_residual_J_kg": (
                                    turbine.isentropic_efficiency_residual_J_kg
                                ),
                            }
                            if turbine.thermo_model_id is not None
                            else {}
                        ),
                    }
                    if hasattr(turbine, "map_id")
                    else {}
                ),
            },
            "nozzle": {
                "choked": nozzle.choked,
                "mass_flow_capacity_kg_s": nozzle.mass_flow_capacity_kg_s,
                "total_pressure_recovery": nozzle.total_pressure_recovery,
                **(
                    {
                        "discharge_coefficient": nozzle.discharge_coefficient,
                        "map_id": nozzle.map_id,
                        "map_version": nozzle.map_version,
                        "map_source_hash": nozzle.map_source_hash,
                        "map_in_domain": nozzle.map_in_domain,
                        "map_extrapolation_distance": (
                            nozzle.map_extrapolation_distance
                        ),
                        "map_uncertainty": dict(nozzle.map_uncertainty),
                        "inlet_pressure_ratio": nozzle.inlet_pressure_ratio,
                        "reynolds_ratio": nozzle.reynolds_ratio,
                        "reynolds_number": nozzle.reynolds_number,
                        "minimum_reynolds_margin": (
                            nozzle.minimum_reynolds_margin
                        ),
                        "maximum_reynolds_margin": (
                            nozzle.maximum_reynolds_margin
                        ),
                        "exit_enthalpy_residual_J_kg": (
                            nozzle.exit_enthalpy_residual_J_kg
                        ),
                        "thermo_model_id": nozzle.thermo_model_id,
                    }
                    if hasattr(nozzle, "map_id")
                    else {}
                ),
            },
        },
        performance={
            "thrust_N": thrust,
            "tsfc_kg_N_s": tsfc,
            "mass_kg": mass.total_mass_kg,
        },
        constraints=all_constraints,
        mass_breakdown_kg=mass.component_masses_kg,
        mass_volume_breakdown_m3=mass.component_volumes_m3,
        formal_diagnostics=formal.diagnostics,
        feasible=feasible,
        core_feasible=thermodynamic_feasible,
        thermodynamic_feasible=thermodynamic_feasible,
        thermodynamic_constraints=thermodynamic_constraints,
        assembly_constraints=geometry.assembly_residuals,
        structural_constraints=structural_diagnostics,
        thermal_constraints=formal.thermal_residuals,
        manufacturing_constraints=formal.manufacturing_residuals,
        assembly_diagnostics=geometry.assembly_residuals,
        structural_diagnostics=structural_diagnostics,
        assembly_diagnostics_pending=False,
        structural_diagnostics_pending=False,
        feasibility_complete=True,
        feasibility_scope="geometry19_wp19_8_bootstrap_complete",
        conservation=ConservationResiduals(
            combustor_energy_W=combustor.energy_residual_W,
            shaft_power_W=turbine.shaft_power_residual_W,
            mass_flow_kg_s=nozzle.mass_flow_residual_kg_s,
            normalized_flow=nozzle.normalized_flow_residual,
            nozzle_velocity_m_s=nozzle.velocity_consistency_residual_m_s,
        ),
        solver=SolverMetadata(
            converged=True,
            flow_residual=nozzle.normalized_flow_residual,
            iterations=flow_match.iterations,
            evaluations=flow_match.evaluations,
            failure_code=None,
            message=(
                "Selected the reference-continuation high-flow branch from "
                f"{len(flow_match.roots)} valid roots"
                if len(flow_match.roots) > 1
                else ""
            ),
        ),
        physics_model_version=model_config.physics_spec_version,
        calibration_pack_hash=model_config.calibration_pack_hash,
        calibration_status=model_config.status,
        formal_dataset_generation_ready=(
            model_config.formal_dataset_generation_ready
        ),
    )
