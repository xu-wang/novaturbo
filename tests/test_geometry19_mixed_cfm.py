import torch

from src.ai.geometry19_mixed_cfm import (
    Geometry19DequantizedCFM,
    labels_to_one_hot,
    normalized_integer_labels,
)


def test_integer_label_round_trip():
    values = torch.tensor([
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    labels = normalized_integer_labels(values, [0, 1, 2], [13, 9, 13])
    assert [item.tolist() for item in labels] == [[0, 12], [4, 0], [12, 6]]
    encoded = labels_to_one_hot(labels, [13, 9, 13])
    assert encoded.shape == (2, 35)
    assert torch.allclose(encoded.sum(dim=1), torch.tensor([3.0, 3.0]))


def test_dequantization_bins_decode_to_exact_integer_grid():
    model = object.__new__(Geometry19DequantizedCFM)
    object.__setattr__(model, "integer_indices", [5, 10, 13])
    object.__setattr__(model, "class_counts", [13, 9, 13])
    values = torch.zeros(4, 19)
    values[:, 5] = torch.tensor([-1.0, -0.99, 0.0, 1.0])
    values[:, 10] = torch.tensor([-0.8, -0.1, 0.7, 0.99])
    values[:, 13] = torch.tensor([-0.5, 0.2, 0.8, 1.0])
    decoded = model._decode_bins(values)
    for index, count in zip(model.integer_indices, model.class_counts, strict=True):
        labels = 0.5 * (decoded[:, index] + 1.0) * (count - 1)
        assert torch.allclose(labels, labels.round())
