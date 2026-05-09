from __future__ import annotations

import torch
from torch import nn


class MLPRisk(nn.Module):
    """A small neural network that maps covariates to one Cox risk score."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
        seed: int | None = 123,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
