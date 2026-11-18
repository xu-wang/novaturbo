#!/usr/bin/env python3
"""Minimal example: predict, evaluate, generate, inverse-design with V2.1."""

from novaturbo_v21 import V21Capability, dumps

capability = V21Capability(device="cpu")
capability.describe()

target = {"thrust_N": 60.0, "tsfc_kg_N_s": 0.000067, "mass_kg": 1.77}
reference = capability.reference_design

print("surrogate prediction:")
print(dumps(capability.predict(reference)))

print("V2.1 physics evaluation:")
print(dumps(capability.evaluate(reference, target=target)["performance"]))

print("generated designs (first 3):")
print(dumps(capability.generate(target, n_samples=8, seed=20260817)[:3]))

result = capability.inverse_design(target, n_samples=256, seed=20260817, physics_budget=16)
print("inverse design status:", result["status"])
print(dumps(result["delivered"]))
