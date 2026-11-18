"""Low-order constant and variable-property thermodynamic backends."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml


class ThermoModelError(ValueError):
    """Raised for invalid thermo metadata, state, or inversion."""


class ThermoModel(Protocol):
    model_id: str
    source_hash: str
    valid_temperature_range_K: tuple[float, float]
    gas_constant_air_J_kg_K: float
    gas_constant_gas_J_kg_K: float

    def cp_air(self, temperature_K: float) -> float: ...
    def cp_gas(self, temperature_K: float, fuel_air_ratio: float) -> float: ...
    def h_air(self, temperature_K: float) -> float: ...
    def h_gas(self, temperature_K: float, fuel_air_ratio: float) -> float: ...
    def air_temperature_from_h(self, enthalpy_J_kg: float) -> float: ...
    def gas_temperature_from_h(
        self, enthalpy_J_kg: float, fuel_air_ratio: float
    ) -> float: ...
    def gamma_air(self, temperature_K: float) -> float: ...
    def gamma_gas(self, temperature_K: float, fuel_air_ratio: float) -> float: ...
    def mu_air(self, temperature_K: float) -> float: ...
    def mu_gas(self, temperature_K: float, fuel_air_ratio: float) -> float: ...
    def air_isentropic_pressure_ratio(
        self, inlet_temperature_K: float, outlet_temperature_K: float
    ) -> float: ...
    def gas_isentropic_pressure_ratio(
        self,
        inlet_temperature_K: float,
        outlet_temperature_K: float,
        fuel_air_ratio: float,
    ) -> float: ...


@dataclass(frozen=True)
class VariablePropertyThermo:
    """Analytic cp polynomials with monotone enthalpy and bisection inverse."""

    model_id: str
    reference_temperature_K: float
    valid_temperature_range_K: tuple[float, float]
    gas_constant_air_J_kg_K: float
    gas_constant_gas_J_kg_K: float
    cp_air_coefficients: tuple[float, float, float]
    cp_gas_coefficients: tuple[float, float, float]
    cp_gas_far_coefficient: float
    viscosity_reference_Pa_s: float
    viscosity_reference_temperature_K: float
    sutherland_constant_K: float
    source_hash: str

    def _temperature(self, value: float) -> float:
        temperature = float(value)
        lower, upper = self.valid_temperature_range_K
        if not math.isfinite(temperature) or not lower <= temperature <= upper:
            raise ThermoModelError(
                f"Temperature {temperature!r} K is outside [{lower}, {upper}]"
            )
        return temperature

    def _cp(self, temperature_K: float, coefficients: tuple[float, float, float]) -> float:
        temperature = self._temperature(temperature_K)
        delta = temperature - self.reference_temperature_K
        value = coefficients[0] + coefficients[1] * delta + coefficients[2] * delta**2
        if not math.isfinite(value) or value <= 0.0:
            raise ThermoModelError("Heat capacity must be finite and positive")
        return value

    def _h(self, temperature_K: float, coefficients: tuple[float, float, float]) -> float:
        temperature = self._temperature(temperature_K)
        delta = temperature - self.reference_temperature_K
        return (
            coefficients[0] * delta
            + 0.5 * coefficients[1] * delta**2
            + coefficients[2] * delta**3 / 3.0
        )

    def cp_air(self, temperature_K: float) -> float:
        return self._cp(temperature_K, self.cp_air_coefficients)

    def cp_gas(self, temperature_K: float, fuel_air_ratio: float) -> float:
        far = float(fuel_air_ratio)
        if not math.isfinite(far) or far < 0.0:
            raise ThermoModelError("Fuel-air ratio must be finite and nonnegative")
        return self._cp(temperature_K, self.cp_gas_coefficients) + (
            self.cp_gas_far_coefficient * far
        )

    def h_air(self, temperature_K: float) -> float:
        return self._h(temperature_K, self.cp_air_coefficients)

    def h_gas(self, temperature_K: float, fuel_air_ratio: float) -> float:
        temperature = self._temperature(temperature_K)
        far = float(fuel_air_ratio)
        if not math.isfinite(far) or far < 0.0:
            raise ThermoModelError("Fuel-air ratio must be finite and nonnegative")
        delta = temperature - self.reference_temperature_K
        return self._h(temperature, self.cp_gas_coefficients) + (
            self.cp_gas_far_coefficient * far * delta
        )

    def air_temperature_from_h(self, enthalpy_J_kg: float) -> float:
        return self._temperature_from_h(
            enthalpy_J_kg, self.cp_air_coefficients, 0.0
        )

    def _temperature_from_h(
        self,
        enthalpy_J_kg: float,
        coefficients: tuple[float, float, float],
        cp_offset: float,
    ) -> float:
        target = float(enthalpy_J_kg)
        if not math.isfinite(target):
            raise ThermoModelError("Enthalpy target must be finite")
        lower, upper = self.valid_temperature_range_K
        def enthalpy(temperature: float) -> float:
            delta = temperature - self.reference_temperature_K
            return self._h(temperature, coefficients) + cp_offset * delta

        if not enthalpy(lower) <= target <= enthalpy(upper):
            raise ThermoModelError("Enthalpy target is outside the thermo model range")
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            if enthalpy(middle) < target:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def gas_temperature_from_h(
        self, enthalpy_J_kg: float, fuel_air_ratio: float
    ) -> float:
        far = float(fuel_air_ratio)
        if not math.isfinite(far) or far < 0.0:
            raise ThermoModelError("Fuel-air ratio must be finite and nonnegative")
        return self._temperature_from_h(
            enthalpy_J_kg,
            self.cp_gas_coefficients,
            self.cp_gas_far_coefficient * far,
        )

    def gamma_air(self, temperature_K: float) -> float:
        cp = self.cp_air(temperature_K)
        return cp / (cp - self.gas_constant_air_J_kg_K)

    def gamma_gas(self, temperature_K: float, fuel_air_ratio: float) -> float:
        cp = self.cp_gas(temperature_K, fuel_air_ratio)
        return cp / (cp - self.gas_constant_gas_J_kg_K)

    def mu_air(self, temperature_K: float) -> float:
        temperature = self._temperature(temperature_K)
        reference = self.viscosity_reference_temperature_K
        value = (
            self.viscosity_reference_Pa_s
            * (temperature / reference) ** 1.5
            * (reference + self.sutherland_constant_K)
            / (temperature + self.sutherland_constant_K)
        )
        if not math.isfinite(value) or value <= 0.0:
            raise ThermoModelError("Dynamic viscosity must be finite and positive")
        return value

    def mu_gas(self, temperature_K: float, fuel_air_ratio: float) -> float:
        far = float(fuel_air_ratio)
        if not math.isfinite(far) or far < 0.0:
            raise ThermoModelError("Fuel-air ratio must be finite and nonnegative")
        # Software-shadow proxy until a validated combustion-product law is supplied.
        return self.mu_air(temperature_K)

    def _isentropic_pressure_ratio(
        self,
        inlet_temperature_K: float,
        outlet_temperature_K: float,
        coefficients: tuple[float, float, float],
        cp_offset: float,
        gas_constant_J_kg_K: float,
    ) -> float:
        inlet = self._temperature(inlet_temperature_K)
        outlet = self._temperature(outlet_temperature_K)
        if outlet == inlet:
            return 1.0
        a, b, c = coefficients
        reference = self.reference_temperature_K
        log_coefficient = a + cp_offset - b * reference + c * reference**2
        entropy_integral = (
            log_coefficient * math.log(outlet / inlet)
            + (b - 2.0 * c * reference) * (outlet - inlet)
            + 0.5 * c * (outlet**2 - inlet**2)
        )
        return math.exp(entropy_integral / gas_constant_J_kg_K)

    def air_isentropic_pressure_ratio(
        self, inlet_temperature_K: float, outlet_temperature_K: float
    ) -> float:
        inlet = self._temperature(inlet_temperature_K)
        outlet = self._temperature(outlet_temperature_K)
        if outlet <= inlet:
            raise ThermoModelError("Compressor isentropic outlet must exceed inlet")
        return self._isentropic_pressure_ratio(
            inlet,
            outlet,
            self.cp_air_coefficients,
            0.0,
            self.gas_constant_air_J_kg_K,
        )

    def gas_isentropic_pressure_ratio(
        self,
        inlet_temperature_K: float,
        outlet_temperature_K: float,
        fuel_air_ratio: float,
    ) -> float:
        far = float(fuel_air_ratio)
        if not math.isfinite(far) or far < 0.0:
            raise ThermoModelError("Fuel-air ratio must be finite and nonnegative")
        return self._isentropic_pressure_ratio(
            inlet_temperature_K,
            outlet_temperature_K,
            self.cp_gas_coefficients,
            self.cp_gas_far_coefficient * far,
            self.gas_constant_gas_J_kg_K,
        )


def _finite_number(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool):
        raise ThermoModelError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermoModelError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ThermoModelError(f"{name} must be finite")
    return result


def load_variable_property_thermo(path: str | Path) -> VariablePropertyThermo:
    source = Path(path).resolve()
    try:
        source_bytes = source.read_bytes()
        payload = yaml.safe_load(source_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise ThermoModelError(f"Cannot load thermo model: {source}") from exc
    required = {
        "model_id",
        "formal_use_allowed",
        "reference_temperature_K",
        "valid_temperature_range_K",
        "gas_constants_J_kg_K",
        "cp_polynomials",
        "viscosity",
        "source",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ThermoModelError("Thermo model metadata keys mismatch")
    model_id = payload["model_id"]
    if not isinstance(model_id, str) or source.stem != model_id:
        raise ThermoModelError("Thermo filename must match non-empty model_id")
    if payload["formal_use_allowed"] is not False:
        raise ThermoModelError("Variable-property shadow thermo must not allow formal use")
    temperature_range = payload["valid_temperature_range_K"]
    if not isinstance(temperature_range, list) or len(temperature_range) != 2:
        raise ThermoModelError("valid_temperature_range_K must have two values")
    lower, upper = (float(item) for item in temperature_range)
    reference = _finite_number(payload, "reference_temperature_K")
    if not 0.0 < lower < reference < upper:
        raise ThermoModelError("Thermo temperature range/reference is invalid")
    gas = payload["gas_constants_J_kg_K"]
    cp = payload["cp_polynomials"]
    viscosity = payload["viscosity"]
    if not isinstance(gas, dict) or set(gas) != {"air", "gas"}:
        raise ThermoModelError("gas_constants_J_kg_K must define air and gas")
    if not isinstance(cp, dict) or set(cp) != {"air", "gas", "gas_far_coefficient"}:
        raise ThermoModelError("cp_polynomials contract mismatch")
    if not isinstance(viscosity, dict) or set(viscosity) != {
        "reference_Pa_s",
        "reference_temperature_K",
        "sutherland_constant_K",
    }:
        raise ThermoModelError("viscosity contract mismatch")
    air_coefficients = tuple(float(item) for item in cp["air"])
    gas_coefficients = tuple(float(item) for item in cp["gas"])
    if len(air_coefficients) != 3 or len(gas_coefficients) != 3:
        raise ThermoModelError("cp polynomials require three coefficients")
    model = VariablePropertyThermo(
        model_id=model_id,
        reference_temperature_K=reference,
        valid_temperature_range_K=(lower, upper),
        gas_constant_air_J_kg_K=_finite_number(gas, "air"),
        gas_constant_gas_J_kg_K=_finite_number(gas, "gas"),
        cp_air_coefficients=air_coefficients,
        cp_gas_coefficients=gas_coefficients,
        cp_gas_far_coefficient=_finite_number(cp, "gas_far_coefficient"),
        viscosity_reference_Pa_s=_finite_number(viscosity, "reference_Pa_s"),
        viscosity_reference_temperature_K=_finite_number(
            viscosity, "reference_temperature_K"
        ),
        sutherland_constant_K=_finite_number(viscosity, "sutherland_constant_K"),
        source_hash=sha256(source_bytes).hexdigest(),
    )
    samples = [lower + index * (upper - lower) / 32.0 for index in range(33)]
    if any(model.cp_air(value) <= model.gas_constant_air_J_kg_K for value in samples):
        raise ThermoModelError("Air cp must exceed its gas constant over the valid range")
    if any(model.cp_gas(value, 0.0) <= model.gas_constant_gas_J_kg_K for value in samples):
        raise ThermoModelError("Gas cp must exceed its gas constant over the valid range")
    return model
