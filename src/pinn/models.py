from __future__ import annotations

import math

import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SIRENLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w0: float, is_first: bool):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.act = Sine(w0)
        self.is_first = is_first
        self.w0 = w0
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                # From SIREN paper
                self.lin.weight.uniform_(-1.0 / self.lin.in_features, 1.0 / self.lin.in_features)
            else:
                bound = math.sqrt(6 / self.lin.in_features) / self.w0
                self.lin.weight.uniform_(-bound, bound)
            if self.lin.bias is not None:
                self.lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin(x))


class SIREN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 128, hidden_layers: int = 4, out_features: int = 1, w0: float = 30.0, w0_hidden: float = 1.0):
        super().__init__()
        layers = [SIRENLayer(in_features, hidden_features, w0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SIRENLayer(hidden_features, hidden_features, w0_hidden, is_first=False))
        self.net = nn.Sequential(*layers, nn.Linear(hidden_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)




