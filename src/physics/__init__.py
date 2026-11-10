"""Physics models and validation utilities for NovaTurbo."""

from src.physics.compressor_meanline import (
    CompressorModelError,
    CompressorResult,
    solve_compressor_meanline,
)
from src.physics.compressor_map_enhanced import (
    MapEnhancedCompressorResult,
    solve_compressor_map_enhanced,
)
from src.physics.combustor_loworder import (
    CombustorModelError,
    CombustorResult,
    solve_combustor_loworder,
)
from src.physics.combustor_map_enhanced import (
    MapEnhancedCombustorResult,
    solve_combustor_map_enhanced,
)
from src.physics.constraint_profiles import (
    ConstraintAssessment,
    ConstraintProfile,
    ConstraintProfileError,
    assess_constraint_profile,
)
from src.physics.engine_match import (
    ConservationResiduals,
    DesignPointTrialState,
    EngineResult,
    Environment,
    FailureCode,
    FlowMatchResult,
    NozzleExitState,
    OperatingPointSelectionPolicy,
    RootSearchConfig,
    SolverMetadata,
    StationState,
    TrialEvaluation,
    find_flow_roots,
    solve_design_point,
)
from src.physics.geometry_state import (
    Geometry19,
    GeometryState,
    GeometryStateError,
    derive_geometry_state,
    geometry19_coordinate_bounds,
)
from src.physics.inlet_map_enhanced import (
    InletModelError,
    MapEnhancedInletState,
    solve_inlet_map_enhanced,
)
from src.physics.formal_constraints_v2 import (
    FormalConstraintError,
    FormalConstraintEvaluation,
    evaluate_formal_constraints_v2,
)
from src.physics.mass_model_v2 import (
    MassBreakdown,
    MassModelError,
    solve_mass_model_v2,
)
from src.physics.model_config import CalibrationPackError, ModelConfig
from src.physics.runtime import (
    PhysicsBackend,
    PhysicsRuntime,
    PhysicsRuntimeError,
    V21PhysicsBackend,
    V22ShadowPhysicsBackend,
    build_physics_runtime,
)
from src.physics.thermo import (
    ThermoModel,
    ThermoModelError,
    VariablePropertyThermo,
    load_variable_property_thermo,
)
from src.physics.nozzle import (
    NozzleModelError,
    NozzleResult,
    compressible_mass_flow_function,
    solve_convergent_nozzle,
)
from src.physics.nozzle_map_enhanced import (
    MapEnhancedNozzleResult,
    solve_nozzle_map_enhanced,
)
from src.physics.single_point_validation_v2 import (
    WP19ValidationError,
    render_wp19_9_markdown,
    run_wp19_9_validation,
    write_wp19_9_artifacts,
)
from src.physics.sanity_doe_v2 import (
    WP19DOEError,
    WP19DOEResult,
    render_wp19_10_markdown,
    run_wp19_10_doe,
    sample_geometry19_designs,
    write_wp19_10_artifacts,
)
from src.physics.turbine_meanline import (
    TurbineModelError,
    TurbineResult,
    solve_turbine_meanline,
)
from src.physics.turbine_map_enhanced import (
    MapEnhancedTurbineResult,
    solve_turbine_map_enhanced,
)

__all__ = [
    "CompressorModelError",
    "CompressorResult",
    "MapEnhancedCompressorResult",
    "CombustorModelError",
    "CombustorResult",
    "MapEnhancedCombustorResult",
    "ConstraintAssessment",
    "ConstraintProfile",
    "ConstraintProfileError",
    "CalibrationPackError",
    "ConservationResiduals",
    "DesignPointTrialState",
    "EngineResult",
    "Environment",
    "FailureCode",
    "FlowMatchResult",
    "FormalConstraintError",
    "FormalConstraintEvaluation",
    "Geometry19",
    "GeometryState",
    "GeometryStateError",
    "MassBreakdown",
    "MassModelError",
    "ModelConfig",
    "InletModelError",
    "MapEnhancedInletState",
    "PhysicsBackend",
    "PhysicsRuntime",
    "PhysicsRuntimeError",
    "NozzleExitState",
    "OperatingPointSelectionPolicy",
    "NozzleModelError",
    "NozzleResult",
    "MapEnhancedNozzleResult",
    "RootSearchConfig",
    "SolverMetadata",
    "StationState",
    "TrialEvaluation",
    "V21PhysicsBackend",
    "V22ShadowPhysicsBackend",
    "ThermoModel",
    "ThermoModelError",
    "VariablePropertyThermo",
    "TurbineModelError",
    "TurbineResult",
    "MapEnhancedTurbineResult",
    "WP19ValidationError",
    "WP19DOEError",
    "WP19DOEResult",
    "derive_geometry_state",
    "assess_constraint_profile",
    "build_physics_runtime",
    "evaluate_formal_constraints_v2",
    "geometry19_coordinate_bounds",
    "render_wp19_9_markdown",
    "render_wp19_10_markdown",
    "run_wp19_9_validation",
    "run_wp19_10_doe",
    "sample_geometry19_designs",
    "compressible_mass_flow_function",
    "find_flow_roots",
    "solve_design_point",
    "solve_mass_model_v2",
    "solve_inlet_map_enhanced",
    "solve_compressor_meanline",
    "solve_compressor_map_enhanced",
    "solve_combustor_loworder",
    "solve_combustor_map_enhanced",
    "solve_turbine_meanline",
    "solve_turbine_map_enhanced",
    "solve_convergent_nozzle",
    "solve_nozzle_map_enhanced",
    "write_wp19_9_artifacts",
    "write_wp19_10_artifacts",
    "load_variable_property_thermo",
]
