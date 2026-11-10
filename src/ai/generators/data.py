"""Shared normalized WP3 dataset and deterministic DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from ..generator_dataset import GeneratorNormalizer, load_schema


class GeneratorDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        schema_path: Path,
        normalizer_path: Path,
        split: str,
    ) -> None:
        schema = load_schema(Path(schema_path))
        normalizer = GeneratorNormalizer.load(Path(normalizer_path))
        frame = pd.read_csv(csv_path)
        frame = frame[(frame["split"] == split) & frame["is_valid"].astype(bool)].copy()
        if frame.empty:
            raise ValueError(f"No valid rows found for split={split}")
        x_names = [field["name"] for field in schema["design_variables"]]
        c_names = [field["name"] for field in schema["conditions"]]
        x = normalizer.transform_design(frame[x_names])
        c = normalizer.transform_conditions(frame[c_names])
        if not np.isfinite(x).all() or not np.isfinite(c).all():
            raise ValueError(f"Non-finite normalized values in split={split}")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.c = torch.as_tensor(c, dtype=torch.float32)
        self.sample_ids = frame["sample_id"].astype(str).tolist()
        self.split = split
        self.schema_version = str(schema["schema_version"])
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "x": self.x[index],
            "condition": self.c[index],
            "sample_id": self.sample_ids[index],
        }


def _deterministic_subset(dataset: Dataset, maximum: Optional[int], seed: int):
    if maximum is None or maximum <= 0 or maximum >= len(dataset):
        return dataset
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=int(maximum), replace=False).tolist()
    return Subset(dataset, indices)


def _dataset_conditions(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        return dataset.dataset.c[torch.as_tensor(dataset.indices, dtype=torch.long)]
    return dataset.c


def build_tail_sampling_weights(
    conditions: torch.Tensor, config: Mapping[str, Any]
) -> torch.Tensor:
    """Return deterministic weights that emphasize condition-space tails."""

    values = torch.as_tensor(conditions, dtype=torch.float64)
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("tail sampling conditions must be a finite 2D tensor")
    quantile = float(config.get("quantile", 0.9))
    maximum_weight = float(config.get("maximum_weight", 4.0))
    power = float(config.get("power", 2.0))
    if not 0.5 <= quantile < 1.0:
        raise ValueError("tail_sampling.quantile must be in [0.5,1)")
    if maximum_weight < 1.0 or power <= 0.0:
        raise ValueError("tail sampling maximum_weight and power are invalid")
    score = values.abs().amax(dim=1)
    threshold = torch.quantile(score, quantile)
    upper = torch.quantile(score, min(0.999, max(quantile + 0.05, 0.99)))
    scale = torch.clamp(upper - threshold, min=torch.finfo(values.dtype).eps)
    tail = torch.clamp((score - threshold) / scale, min=0.0, max=1.0)
    return 1.0 + (maximum_weight - 1.0) * tail.pow(power)


def build_generator_dataloaders(
    data_config: Mapping[str, Any],
    repo_root: Path,
    loader_seed: int,
) -> Tuple[DataLoader, DataLoader, GeneratorNormalizer, str, Dict[str, int]]:
    dataset_path = (repo_root / data_config["dataset_csv"]).resolve()
    schema_path = (repo_root / data_config["schema"]).resolve()
    normalizer_path = (repo_root / data_config["normalizer"]).resolve()
    train_full = GeneratorDataset(dataset_path, schema_path, normalizer_path, "train")
    val_full = GeneratorDataset(dataset_path, schema_path, normalizer_path, "val")
    subset_seed = int(data_config.get("subset_seed", 314159))
    train = _deterministic_subset(
        train_full, data_config.get("max_train_samples"), subset_seed
    )
    val = _deterministic_subset(
        val_full, data_config.get("max_val_samples"), subset_seed + 1
    )
    generator = torch.Generator().manual_seed(int(loader_seed))
    batch_size = int(data_config.get("batch_size", 512))
    tail_config = data_config.get("tail_sampling", {})
    tail_enabled = bool(
        tail_config is True
        or (isinstance(tail_config, Mapping) and tail_config.get("enabled", False))
    )
    sampler = None
    if tail_enabled:
        if not isinstance(tail_config, Mapping):
            tail_config = {"enabled": True}
        weights = build_tail_sampling_weights(
            _dataset_conditions(train), tail_config
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=int(data_config.get("num_workers", 0)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        drop_last=False,
    )
    counts = {
        "train_full": len(train_full),
        "val_full": len(val_full),
        "train_used": len(train),
        "val_used": len(val),
        "tail_sampling_enabled": tail_enabled,
    }
    return (
        train_loader,
        val_loader,
        train_full.normalizer,
        train_full.schema_version,
        counts,
    )
