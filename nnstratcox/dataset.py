from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class StratifiedRiskSetDataset(Dataset):
    """Event-centered risk sets for batched stratified Cox training.

    One item is one Cox likelihood component:

    - the observed event subject is labeled 1
    - all at-risk subjects in the same stratum are labeled 0 for this component
    - the risk set is therefore ``duration_j >= duration_case`` and
      ``strata_j == strata_case``

    A DataLoader batch contains multiple such components. This mirrors the
    stratum-batch style used in the larger project, but uses full same-stratum
    risk sets rather than sampled NCC controls.
    """

    def __init__(
        self,
        x: np.ndarray,
        duration: np.ndarray,
        event: np.ndarray,
        strata: np.ndarray,
        max_controls: int | None = None,
        seed: int = 123,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.duration = np.asarray(duration, dtype=np.float32)
        self.event = np.asarray(event, dtype=np.float32)
        self.strata = np.asarray(strata)
        self.max_controls = max_controls
        self.rng = np.random.default_rng(seed)

        if not (self.x.shape[0] == self.duration.shape[0] == self.event.shape[0] == self.strata.shape[0]):
            raise ValueError("x, duration, event, and strata must have the same number of rows")

        self.case_indices = np.where(self.event == 1)[0].astype(np.int64)
        if self.case_indices.size == 0:
            raise ValueError("Cannot build risk sets: no observed events")

    def __len__(self) -> int:
        return int(self.case_indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case_idx = int(self.case_indices[item])
        same_stratum = self.strata == self.strata[case_idx]
        at_risk = self.duration >= self.duration[case_idx]
        risk_idx = np.where(same_stratum & at_risk)[0].astype(np.int64)

        controls = risk_idx[risk_idx != case_idx]
        if self.max_controls is not None and controls.size > self.max_controls:
            controls = self.rng.choice(controls, size=self.max_controls, replace=False).astype(np.int64)
        idx = np.concatenate([np.array([case_idx], dtype=np.int64), controls])

        # Sort so the Cox suffix-sum implementation can build risk sets by time.
        order = np.argsort(self.duration[idx], kind="mergesort")
        idx = idx[order]

        local_event = np.zeros(idx.shape[0], dtype=np.float32)
        local_event[np.where(idx == case_idx)[0][0]] = 1.0

        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.duration[idx].astype(np.float32)),
            torch.from_numpy(local_event),
        )


def risk_set_collate(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    xs, durations, events = [], [], []
    for x, duration, event in batch:
        xs.append(x)
        durations.append(duration)
        events.append(event)
    return xs, durations, events
