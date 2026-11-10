"""Three-performance gate and plain-CFM models; no physics or target access."""
import math

import torch
from torch import nn


def condition3(condition):
    if condition.ndim != 2 or condition.shape[1] != 3:
        raise ValueError("External condition must contain exactly three performances")


class PerformanceGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 33))

    def forward(self, condition):
        condition3(condition)
        return self.net(condition)


class FiLMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128, elementwise_affine=False)
        self.film = nn.Linear(96, 256)
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))

    def forward(self, hidden, context):
        scale, shift = self.film(context).chunk(2, dim=-1)
        return hidden + self.mlp(torch.nn.functional.silu(self.norm(hidden) * (1 + scale) + shift))


class FlowTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition_encoder = nn.Sequential(nn.Linear(3, 64), nn.SiLU())
        self.input = nn.Linear(16, 128)
        self.blocks = nn.ModuleList([FiLMBlock(), FiLMBlock()])
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 16)))

    def forward(self, state, time, condition):
        condition3(condition)
        if state.shape != (len(condition), 16) or time.shape != (len(condition),):
            raise ValueError("Expected aligned state [N,16], time [N], condition [N,3]")
        phase = time[:, None] * self.frequencies[None, :] * (2 * math.pi)
        context = torch.cat((self.condition_encoder(condition), phase.sin(), phase.cos()), dim=-1)
        hidden = self.input(state)
        for block in self.blocks:
            hidden = block(hidden, context)
        return hidden


def flow_head():
    return nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 16))


class LocalCFM(nn.Module):
    """Dispatch IDs are supervision/internal routing, never encoded as a condition."""
    def __init__(self):
        super().__init__()
        self.trunk = FlowTrunk()
        self.heads = nn.ModuleList([flow_head() for _ in range(33)])

    def forward(self, state, time, condition, dispatch):
        if dispatch.shape != (len(condition),) or dispatch.dtype != torch.long:
            raise ValueError("Expected one integer dispatch index per row")
        if bool(((dispatch < 0) | (dispatch >= 33)).any()):
            raise ValueError("Dispatch index outside 33 frozen experts")
        hidden = self.trunk(state, time, condition)
        result = torch.zeros_like(state)
        for index, head in enumerate(self.heads):
            rows = torch.nonzero(dispatch == index, as_tuple=True)[0]
            if rows.numel():
                result = result.index_copy(0, rows, head(hidden.index_select(0, rows)))
        return result


class PooledCFM(nn.Module):
    def __init__(self, category_counts):
        super().__init__()
        if len(category_counts) != 3 or any(n < 2 for n in category_counts):
            raise ValueError("Expected three schema categorical axes")
        self.trunk = FlowTrunk()
        self.head = flow_head()
        self.integer_encoder = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.integer_heads = nn.ModuleList([nn.Linear(64, n) for n in category_counts])

    def forward(self, state, time, condition):
        return self.head(self.trunk(state, time, condition))

    def integer_logits(self, condition):
        condition3(condition)
        hidden = self.integer_encoder(condition)
        return [head(hidden) for head in self.integer_heads]
