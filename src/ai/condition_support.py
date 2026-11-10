"""Calibrated support diagnostics for normalized generator conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ConditionSupportModel:
    train_conditions: np.ndarray
    neighbor_count: int
    distance_threshold: float
    max_abs_threshold: float

    @classmethod
    def fit(
        cls,
        train_conditions: np.ndarray,
        calibration_conditions: np.ndarray,
        *,
        neighbor_count: int = 5,
        quantile: float = 0.99,
    ) -> "ConditionSupportModel":
        train = np.asarray(train_conditions, dtype=float)
        calibration = np.asarray(calibration_conditions, dtype=float)
        if train.ndim != 2 or calibration.ndim != 2 or train.shape[1] != calibration.shape[1]:
            raise ValueError("condition support arrays must be aligned 2D matrices")
        if not np.isfinite(train).all() or not np.isfinite(calibration).all():
            raise ValueError("condition support arrays must be finite")
        if neighbor_count <= 0 or neighbor_count > len(train):
            raise ValueError("neighbor_count must fit within the training set")
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile must be in (0.5,1)")
        tree = cKDTree(train)
        distances, _ = tree.query(calibration, k=neighbor_count)
        kth = distances if neighbor_count == 1 else distances[:, -1]
        return cls(
            train_conditions=train,
            neighbor_count=int(neighbor_count),
            distance_threshold=float(np.quantile(kth, quantile)),
            max_abs_threshold=float(np.quantile(np.abs(calibration).max(axis=1), quantile)),
        )

    def diagnose(self, condition: np.ndarray) -> Dict[str, Any]:
        value = np.asarray(condition, dtype=float).reshape(-1)
        if value.shape != (self.train_conditions.shape[1],) or not np.isfinite(value).all():
            raise ValueError("condition has the wrong shape or contains non-finite values")
        distances, _ = cKDTree(self.train_conditions).query(
            value, k=self.neighbor_count
        )
        kth = float(distances if self.neighbor_count == 1 else distances[-1])
        maximum = float(np.abs(value).max())
        distance_out = kth > self.distance_threshold
        magnitude_out = maximum > self.max_abs_threshold
        return {
            "condition_kth_neighbor_distance": kth,
            "condition_distance_threshold": self.distance_threshold,
            "condition_max_abs": maximum,
            "condition_max_abs_threshold": self.max_abs_threshold,
            "condition_distance_out_of_support": bool(distance_out),
            "condition_magnitude_out_of_support": bool(magnitude_out),
            "condition_support_status": (
                "out_of_support" if distance_out or magnitude_out else "in_support"
            ),
        }
