"""
NovaTurbo — Multi-Objective Optimizer (NSGA-II)

Optimizes engine design for multiple competing objectives:
- Maximize thrust
- Minimize fuel consumption (TSFC)
- Minimize weight
- Maximize thrust-to-weight ratio

Uses NSGA-II genetic algorithm to find the Pareto-optimal front
of non-dominated designs.

Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from ..physics.brayton import FlightConditions, EngineInputs, solve_brayton_cycle
from .dataset import DesignSpaceBounds, evaluate_design

try:
    import pandas as pd
except ImportError:
    pd = None


TARGET_WEIGHTED_RELATIVE_ERROR = "target_weighted_relative_error"


@dataclass
class OptimizationConfig:
    """Configuration for NSGA-II optimizer."""
    # GA parameters
    population_size: int = 100
    n_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1        # Gaussian mutation std (fraction of range)

    # Tournament selection
    tournament_size: int = 3

    # Objectives (minimize all — negate for maximization)
    objectives: List[str] = field(default_factory=lambda: [
        'thrust_N',            # Maximize (negated)
        'tsfc_kg_N_s',         # Minimize
        'total_mass_kg',       # Minimize
        'thrust_to_weight',    # Maximize (negated)
    ])
    maximize: List[bool] = field(default_factory=lambda: [
        True,   # thrust
        False,  # TSFC
        False,  # mass
        True,   # T/W
    ])

    # Optional target-matching objective.  When ``objectives`` contains
    # ``target_weighted_relative_error``, these frozen values define the
    # performance point that the optimizer must approach.  The legacy raw
    # maximize/minimize path is unchanged when the mapping is empty.
    target_values: Dict[str, float] = field(default_factory=dict)
    target_weights: Dict[str, float] = field(default_factory=dict)
    target_epsilons: Dict[str, float] = field(default_factory=dict)

    # Constraints
    min_thrust_N: float = 30.0
    max_tsfc: float = 0.00005
    max_mass_kg: float = 3.0

    # Use surrogate model for fast evaluation?
    use_surrogate: bool = False
    surrogate_model_path: Optional[str] = None

    # Explicit Warm-Start handling.  These values are intentionally ignored by
    # the legacy path when neither initial_genes nor initializer is supplied.
    warm_start_seed: int = 42
    warm_start_fill_method: str = "random"  # random or lhs
    warm_start_out_of_bounds_policy: str = "reject"  # reject or clip
    warm_start_duplicate_tolerance: float = 1.0e-9
    # Advance the global RNG exactly as the WP0 random initializer would have,
    # so Random and Warm-Start runs share the same downstream genetic RNG stream.
    warm_start_align_legacy_rng: bool = True


@dataclass
class Individual:
    """A single design in the population."""
    genes: np.ndarray = field(default_factory=lambda: np.array([]))
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = 0
    crowding_distance: float = 0.0
    is_feasible: bool = True
    performance: dict = field(default_factory=dict)
    individual_id: str = ""
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class InitializationContext:
    """Read-only optimizer schema passed to a custom initializer."""

    param_names: Tuple[str, ...]
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray
    population_size: int


@dataclass
class InitialPopulation:
    """Genes and row-level provenance returned by an initializer.

    ``apply_integer_projection`` defaults to true for arbitrary external 19D
    arrays.  A named 5D adapter may turn it off only when it can prove that no
    integer field was generated and its background follows a frozen baseline.
    """

    genes: np.ndarray
    provenance: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    apply_integer_projection: bool = True


InitializerOutput = Union[np.ndarray, InitialPopulation]
Initializer = Callable[[InitializationContext], InitializerOutput]


INTEGER_PARAMETER_NAMES = (
    "compressor_blade_count",
    "combustor_num_injectors",
    "turbine_blade_count",
)


def optimizer_parameter_space(
    bounds: DesignSpaceBounds,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return the unique ordered 19D schema used by NSGA-II."""

    param_names = list(bounds.__dataclass_fields__.keys())
    if len(param_names) != len(set(param_names)):
        raise ValueError("DesignSpaceBounds contains duplicate parameter names")
    bounds_lo = np.asarray([getattr(bounds, name)[0] for name in param_names], dtype=float)
    bounds_hi = np.asarray([getattr(bounds, name)[1] for name in param_names], dtype=float)
    if np.any(~np.isfinite(bounds_lo)) or np.any(~np.isfinite(bounds_hi)):
        raise ValueError("Optimizer bounds contain non-finite values")
    if np.any(bounds_hi <= bounds_lo):
        raise ValueError("Every optimizer upper bound must exceed its lower bound")
    return param_names, bounds_lo, bounds_hi


def _sample_fill_population(
    count: int,
    n_params: int,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, n_params), dtype=float)
    rng = np.random.RandomState(seed)
    if method == "random":
        unit = rng.random((count, n_params))
    elif method == "lhs":
        unit = np.zeros((count, n_params), dtype=float)
        intervals = np.linspace(0.0, 1.0, count + 1)
        for dimension in range(n_params):
            points = intervals[:-1] + rng.uniform(0.0, 1.0, count) / count
            rng.shuffle(points)
            unit[:, dimension] = points
    else:
        raise ValueError("warm_start_fill_method must be 'random' or 'lhs'")
    return bounds_lo + unit * (bounds_hi - bounds_lo)


def _coerce_initializer_output(
    output: InitializerOutput,
    source: str,
) -> InitialPopulation:
    if isinstance(output, InitialPopulation):
        return output
    return InitialPopulation(
        genes=np.asarray(output),
        metadata={"source": source},
        apply_integer_projection=True,
    )


def _prepare_external_initial_population(
    initial: InitialPopulation,
    param_names: List[str],
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    config: OptimizationConfig,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    genes = np.asarray(initial.genes, dtype=float)
    n_params = len(param_names)
    if genes.ndim != 2:
        raise ValueError(f"initial genes must be a 2D array, got shape {genes.shape}")
    if genes.shape[1] != n_params:
        raise ValueError(
            f"initial genes must have {n_params} columns in optimizer schema order, "
            f"got {genes.shape[1]}"
        )
    if not np.isfinite(genes).all():
        locations = np.argwhere(~np.isfinite(genes))[:8].tolist()
        raise ValueError(f"initial genes contain NaN/inf at indices {locations}")
    genes = genes.copy()
    input_count = len(genes)
    if initial.provenance is None:
        provenance = [
            {
                "source": str(initial.metadata.get("source", "initial_genes")),
                "source_row": index,
            }
            for index in range(input_count)
        ]
    else:
        if len(initial.provenance) != input_count:
            raise ValueError(
                "InitialPopulation provenance length must equal its gene row count"
            )
        provenance = [dict(value) for value in initial.provenance]

    below = genes < bounds_lo[None, :]
    above = genes > bounds_hi[None, :]
    out_of_bounds = below | above
    out_of_bounds_value_count = int(out_of_bounds.sum())
    policy = str(config.warm_start_out_of_bounds_policy)
    if out_of_bounds_value_count:
        if policy == "reject":
            locations = np.argwhere(out_of_bounds)[:8].tolist()
            raise ValueError(
                "initial genes are outside DesignSpaceBounds at indices "
                f"{locations}; set warm_start_out_of_bounds_policy='clip' "
                "only for an explicitly labelled experiment"
            )
        if policy != "clip":
            raise ValueError(
                "warm_start_out_of_bounds_policy must be 'reject' or 'clip'"
            )
        genes = np.clip(genes, bounds_lo, bounds_hi)
    elif policy not in {"reject", "clip"}:
        raise ValueError("warm_start_out_of_bounds_policy must be 'reject' or 'clip'")

    projected_value_count = 0
    if initial.apply_integer_projection:
        for name in INTEGER_PARAMETER_NAMES:
            if name not in param_names:
                continue
            index = param_names.index(name)
            projected = np.clip(np.rint(genes[:, index]), bounds_lo[index], bounds_hi[index])
            projected_value_count += int(np.count_nonzero(projected != genes[:, index]))
            genes[:, index] = projected

    tolerance = float(config.warm_start_duplicate_tolerance)
    if tolerance < 0.0:
        raise ValueError("warm_start_duplicate_tolerance must be non-negative")
    scale = bounds_hi - bounds_lo
    normalized = (genes - bounds_lo[None, :]) / scale[None, :]
    keep_indices: List[int] = []
    duplicate_indices: List[int] = []
    for index, row in enumerate(normalized):
        if any(
            np.max(np.abs(row - normalized[prior])) <= tolerance
            for prior in keep_indices
        ):
            duplicate_indices.append(index)
        else:
            keep_indices.append(index)
    genes = genes[keep_indices]
    provenance = [provenance[index] for index in keep_indices]

    truncated_count = max(0, len(genes) - config.population_size)
    if truncated_count:
        genes = genes[: config.population_size]
        provenance = provenance[: config.population_size]

    fill_count = config.population_size - len(genes)
    fill_method = str(config.warm_start_fill_method)
    fill_seed = int(config.warm_start_seed)
    if fill_count:
        fill = _sample_fill_population(
            fill_count,
            n_params,
            bounds_lo,
            bounds_hi,
            fill_method,
            fill_seed,
        )
        # Explicit Warm-Start populations are made type-valid at generation 0.
        for name in INTEGER_PARAMETER_NAMES:
            if name in param_names:
                index = param_names.index(name)
                fill[:, index] = np.rint(fill[:, index])
        genes = np.vstack([genes, fill])
        provenance.extend(
            {
                "source": f"{fill_method}_fill",
                "source_row": index,
                "fill_seed": fill_seed,
            }
            for index in range(fill_count)
        )

    for index, record in enumerate(provenance):
        record["population_index"] = index
    metadata = {
        "mode": str(initial.metadata.get("mode", "external")),
        "input_count": input_count,
        "unique_count_before_truncation": len(keep_indices),
        "duplicate_count": len(duplicate_indices),
        "duplicate_source_rows": duplicate_indices,
        "out_of_bounds_value_count": out_of_bounds_value_count,
        "out_of_bounds_policy": policy,
        "integer_projection_value_count": projected_value_count,
        "truncated_count": truncated_count,
        "fill_count": fill_count,
        "fill_method": fill_method if fill_count else None,
        "fill_seed": fill_seed if fill_count else None,
        "population_size": len(genes),
        **dict(initial.metadata),
    }
    return genes, provenance, metadata


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if solution a dominates solution b (all objectives minimized)."""
    return np.all(a <= b) and np.any(a < b)


def _fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """NSGA-II fast non-dominated sorting."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]  # Remove empty fronts


def _crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    """Calculate crowding distance for a Pareto front."""
    n = len(front)
    distances = np.zeros(n)

    if n <= 2:
        distances[:] = float('inf')
        return distances

    n_obj = objectives.shape[1]
    for m in range(n_obj):
        # Sort by objective m
        sorted_idx = np.argsort(objectives[front, m])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        obj_range = objectives[front[sorted_idx[-1]], m] - objectives[front[sorted_idx[0]], m]
        if obj_range < 1e-10:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                objectives[front[sorted_idx[i + 1]], m] -
                objectives[front[sorted_idx[i - 1]], m]
            ) / obj_range

    return distances


def _tournament_select(population: List[Individual], k: int = 3) -> Individual:
    """Tournament selection based on rank and crowding distance."""
    candidates = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = candidates[0]
    for c in candidates[1:]:
        if (population[c].rank < population[best].rank or
            (population[c].rank == population[best].rank and
             population[c].crowding_distance > population[best].crowding_distance)):
            best = c
    return population[best]


def _sbx_crossover(parent1: np.ndarray, parent2: np.ndarray,
                   bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                   eta: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX)."""
    n = len(parent1)
    child1, child2 = parent1.copy(), parent2.copy()

    for i in range(n):
        if np.random.random() < 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-10:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

                child1[i] = np.clip(child1[i], bounds_lo[i], bounds_hi[i])
                child2[i] = np.clip(child2[i], bounds_lo[i], bounds_hi[i])

    return child1, child2


def _polynomial_mutation(genes: np.ndarray, bounds_lo: np.ndarray,
                         bounds_hi: np.ndarray, rate: float,
                         eta: float = 20.0) -> np.ndarray:
    """Polynomial mutation."""
    mutated = genes.copy()
    for i in range(len(genes)):
        if np.random.random() < rate:
            delta = bounds_hi[i] - bounds_lo[i]
            if delta < 1e-10:
                continue
            u = np.random.random()
            if u < 0.5:
                delta_q = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            mutated[i] = genes[i] + delta_q * delta
            mutated[i] = np.clip(mutated[i], bounds_lo[i], bounds_hi[i])
    return mutated


def _population_evaluation_records(
    population: List[Individual],
    param_names: List[str],
    generation: int,
    start_index: int,
) -> List[Dict[str, Any]]:
    """Create a complete physics-evaluation ledger without changing evaluation."""

    records: List[Dict[str, Any]] = []
    for offset, individual in enumerate(population):
        record: Dict[str, Any] = {
            "evaluation_index": start_index + offset,
            "generation": generation,
            "evaluation_stage": "initial" if generation == 0 else "offspring",
            "individual_id": individual.individual_id,
            "source": individual.provenance.get("source", "unknown"),
            "parents": list(individual.provenance.get("parents", [])),
            "is_feasible": bool(individual.is_feasible),
        }
        record.update(
            {
                f"gene__{name}": float(individual.genes[index])
                for index, name in enumerate(param_names)
            }
        )
        record.update(
            {
                f"objective_internal_{index}": float(value)
                for index, value in enumerate(individual.objectives)
            }
        )
        record.update(
            {
                f"performance__{name}": value
                for name, value in individual.performance.items()
            }
        )
        records.append(record)
    return records


def optimize(
    config: Optional[OptimizationConfig] = None,
    bounds: Optional[DesignSpaceBounds] = None,
    verbose: bool = True,
    initial_genes: Optional[Union[np.ndarray, InitialPopulation]] = None,
    initializer: Optional[Initializer] = None,
) -> Tuple[List[Individual], dict]:
    """
    Run NSGA-II multi-objective optimization.

    Returns:
        (pareto_front, optimization_history)
    """
    if config is None:
        config = OptimizationConfig()
    if bounds is None:
        bounds = DesignSpaceBounds()

    if len(config.objectives) != len(config.maximize):
        raise ValueError("objectives and maximize must have equal lengths")
    if TARGET_WEIGHTED_RELATIVE_ERROR in config.objectives:
        target_index = config.objectives.index(TARGET_WEIGHTED_RELATIVE_ERROR)
        if config.maximize[target_index]:
            raise ValueError("target_weighted_relative_error must be minimized")
        if not config.target_values:
            raise ValueError("target_values are required for target-directed optimization")
        if set(config.target_values) != set(config.target_weights):
            raise ValueError("target_values and target_weights must have identical keys")
        if any(
            not np.isfinite(float(value)) or float(value) <= 0.0
            for value in config.target_values.values()
        ):
            raise ValueError("target_values must be positive and finite")
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in config.target_weights.values()
        ) or sum(float(value) for value in config.target_weights.values()) <= 0.0:
            raise ValueError("target_weights must be finite, non-negative, and non-zero")

    if initial_genes is not None and initializer is not None:
        raise ValueError("initial_genes and initializer are mutually exclusive")

    # Extract bounds as arrays
    param_names, bounds_lo, bounds_hi = optimizer_parameter_space(bounds)
    n_params = len(param_names)

    if verbose:
        print(f"NovaTurbo NSGA-II Optimizer")
        print(f"  Population: {config.population_size}")
        print(f"  Generations: {config.n_generations}")
        print(f"  Parameters: {n_params}")
        print(f"  Objectives: {config.objectives}")

    # Initialize population.  The no-argument branch intentionally retains the
    # exact WP0 loop and global NumPy RNG consumption order.
    population: List[Individual] = []
    if initial_genes is None and initializer is None:
        initialization_metadata: Dict[str, Any] = {
            "mode": "legacy_random",
            "population_size": config.population_size,
            "wp0_compatible": True,
        }
        initialization_provenance = []
        initial_matrix_rows = []
        for index in range(config.population_size):
            genes = bounds_lo + np.random.random(n_params) * (bounds_hi - bounds_lo)
            record = {
                "population_index": index,
                "source": "legacy_random",
            }
            ind = Individual(
                genes=genes,
                individual_id=f"initial_{index:05d}",
                provenance=record,
            )
            population.append(ind)
            initialization_provenance.append(record)
            initial_matrix_rows.append(genes.copy())
        initial_matrix = np.asarray(initial_matrix_rows, dtype=float)
    else:
        if initializer is not None:
            context = InitializationContext(
                param_names=tuple(param_names),
                bounds_lo=bounds_lo.copy(),
                bounds_hi=bounds_hi.copy(),
                population_size=config.population_size,
            )
            supplied = _coerce_initializer_output(
                initializer(context), source="initializer"
            )
            supplied.metadata.setdefault("interface", "initializer")
        else:
            supplied = _coerce_initializer_output(
                initial_genes, source="initial_genes"  # type: ignore[arg-type]
            )
            supplied.metadata.setdefault("interface", "initial_genes")
        initial_matrix, initialization_provenance, initialization_metadata = (
            _prepare_external_initial_population(
                supplied, param_names, bounds_lo, bounds_hi, config
            )
        )
        for index, genes in enumerate(initial_matrix):
            record = dict(initialization_provenance[index])
            population.append(
                Individual(
                    genes=genes.copy(),
                    individual_id=f"initial_{index:05d}",
                    provenance=record,
                )
            )
        if bool(config.warm_start_align_legacy_rng):
            np.random.random((config.population_size, n_params))
            initialization_metadata["legacy_rng_alignment_draw_count"] = (
                config.population_size * n_params
            )
        else:
            initialization_metadata["legacy_rng_alignment_draw_count"] = 0

    # Evaluate initial population
    _evaluate_population(population, param_names, config)
    evaluation_ledger = _population_evaluation_records(
        population, param_names, generation=0, start_index=0
    )

    history = {
        'best_objectives': [],
        'pareto_size': [],
        'feasible_count': [],
        'initialization': initialization_metadata,
        'initial_population_genes': initial_matrix.tolist(),
        'initial_population_provenance': initialization_provenance,
        'param_names': param_names,
        'evaluation_ledger': evaluation_ledger,
    }

    t0 = time.time()
    for gen in range(config.n_generations):
        # Create offspring
        offspring = []
        while len(offspring) < config.population_size:
            p1 = _tournament_select(population, config.tournament_size)
            p2 = _tournament_select(population, config.tournament_size)

            if np.random.random() < config.crossover_rate:
                c1_genes, c2_genes = _sbx_crossover(
                    p1.genes, p2.genes, bounds_lo, bounds_hi)
            else:
                c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()

            c1_genes = _polynomial_mutation(c1_genes, bounds_lo, bounds_hi, config.mutation_rate)
            c2_genes = _polynomial_mutation(c2_genes, bounds_lo, bounds_hi, config.mutation_rate)

            first_index = len(offspring)
            offspring.append(
                Individual(
                    genes=c1_genes,
                    individual_id=f"generation_{gen + 1:05d}_{first_index:05d}",
                    provenance={
                        "source": "offspring",
                        "generation": gen + 1,
                        "parents": [p1.individual_id, p2.individual_id],
                    },
                )
            )
            offspring.append(
                Individual(
                    genes=c2_genes,
                    individual_id=f"generation_{gen + 1:05d}_{first_index + 1:05d}",
                    provenance={
                        "source": "offspring",
                        "generation": gen + 1,
                        "parents": [p1.individual_id, p2.individual_id],
                    },
                )
            )

        offspring = offspring[:config.population_size]

        # Evaluate offspring
        _evaluate_population(offspring, param_names, config)
        history['evaluation_ledger'].extend(
            _population_evaluation_records(
                offspring,
                param_names,
                generation=gen + 1,
                start_index=len(history['evaluation_ledger']),
            )
        )

        # Combine parent + offspring
        combined = population + offspring

        # Extract objectives
        all_objectives = np.array([ind.objectives for ind in combined])

        # Non-dominated sorting
        fronts = _fast_non_dominated_sort(all_objectives)

        # Assign ranks
        for rank, front in enumerate(fronts):
            for idx in front:
                combined[idx].rank = rank

        # Select next generation
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= config.population_size:
                # Add entire front
                for idx in front:
                    new_population.append(combined[idx])
            else:
                # Partial front — use crowding distance
                cd = _crowding_distance(all_objectives, front)
                sorted_by_cd = np.argsort(-cd)  # Descending
                remaining = config.population_size - len(new_population)
                for i in range(remaining):
                    idx = front[sorted_by_cd[i]]
                    combined[idx].crowding_distance = cd[sorted_by_cd[i]]
                    new_population.append(combined[idx])
                break

        population = new_population

        # Track history
        pareto_front = [ind for ind in population if ind.rank == 0]
        feasible = [ind for ind in population if ind.is_feasible]
        history['pareto_size'].append(len(pareto_front))
        history['feasible_count'].append(len(feasible))

        if pareto_front:
            best_obj = np.array([ind.objectives for ind in pareto_front])
            history['best_objectives'].append(best_obj.mean(axis=0).tolist())

        if verbose and (gen + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Gen {gen+1:4d}/{config.n_generations}  "
                  f"Pareto: {len(pareto_front):3d}  "
                  f"Feasible: {len(feasible):3d}  "
                  f"Time: {elapsed:.1f}s")

    # Final Pareto front
    pareto_front = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda x: x.objectives[0]  # Sort by first objective
    )

    if verbose:
        print(f"\n  Optimization complete!")
        print(f"  Pareto front size: {len(pareto_front)}")
        print(f"  Total time: {time.time()-t0:.1f}s")

        if pareto_front:
            print(f"\n  Pareto Front Summary:")
            for i, obj_name in enumerate(config.objectives):
                vals = [ind.objectives[i] for ind in pareto_front]
                sign = -1 if config.maximize[i] else 1
                print(f"    {obj_name:25s}: {min(v*sign for v in vals):.4f} — "
                      f"{max(v*sign for v in vals):.4f}")

    return pareto_front, history


def _evaluate_population(population: List[Individual],
                         param_names: List[str],
                         config: OptimizationConfig):
    """Evaluate all individuals in the population."""
    for ind in population:
        # Convert genes to parameter dict
        params = {}
        for i, name in enumerate(param_names):
            params[name] = ind.genes[i]

        # Create a pandas-like object for evaluate_design
        if pd is not None:
            row = pd.Series(params)
        else:
            row = type('Row', (), params)()

        result = evaluate_design(row)
        ind.performance = result
        ind.is_feasible = result.get('is_valid', False)

        # Extract objectives (negate for maximization).  Target matching is a
        # scalar minimization problem: zero means the requested performance was
        # reproduced exactly after the physics evaluation.
        objectives = []
        for i, obj_name in enumerate(config.objectives):
            if obj_name == TARGET_WEIGHTED_RELATIVE_ERROR:
                weighted_error = 0.0
                weight_sum = 0.0
                for performance_name, target_value in config.target_values.items():
                    actual = float(result.get(performance_name, np.nan))
                    epsilon = float(config.target_epsilons.get(performance_name, 1.0e-15))
                    relative_error = abs(actual - float(target_value)) / max(
                        abs(float(target_value)), epsilon
                    )
                    result[f"target_relative_error__{performance_name}"] = relative_error
                    weight = float(config.target_weights[performance_name])
                    weighted_error += weight * relative_error
                    weight_sum += weight
                val = weighted_error / weight_sum
                result[TARGET_WEIGHTED_RELATIVE_ERROR] = val
            else:
                val = result.get(obj_name, 0)
            if config.maximize[i]:
                val = -val  # Negate for minimization framework
            objectives.append(val)

        # Penalty for infeasible designs
        if not ind.is_feasible:
            objectives = [1e10] * len(objectives)

        ind.objectives = np.array(objectives)


def get_best_design(pareto_front: List[Individual],
                    priority: str = "thrust_to_weight") -> Optional[Individual]:
    """
    Select the best design from the Pareto front based on a priority metric.
    """
    if not pareto_front:
        return None

    feasible = [ind for ind in pareto_front if ind.is_feasible]
    if not feasible:
        return pareto_front[0]

    if priority == "thrust_to_weight":
        return max(feasible, key=lambda x: x.performance.get('thrust_to_weight', 0))
    elif priority == "thrust":
        return max(feasible, key=lambda x: x.performance.get('thrust_N', 0))
    elif priority == "efficiency":
        return min(feasible, key=lambda x: x.performance.get('tsfc_kg_N_s', float('inf')))
    elif priority == "lightweight":
        return min(feasible, key=lambda x: x.performance.get('total_mass_kg', float('inf')))
    else:
        return feasible[0]


if __name__ == "__main__":
    # Quick test with small population
    config = OptimizationConfig(population_size=20, n_generations=10)
    pareto, history = optimize(config, verbose=True)

    best = get_best_design(pareto, priority="thrust_to_weight")
    if best:
        print(f"\nBest design (T/W priority):")
        for k, v in best.performance.items():
            print(f"  {k}: {v}")
