from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurvivalData:
    x: np.ndarray
    duration: np.ndarray
    event: np.ndarray
    center: np.ndarray
    true_log_risk: np.ndarray
    feature_names: list[str]

    def subset(self, idx: np.ndarray) -> "SurvivalData":
        return SurvivalData(
            x=self.x[idx],
            duration=self.duration[idx],
            event=self.event[idx],
            center=self.center[idx],
            true_log_risk=self.true_log_risk[idx],
            feature_names=self.feature_names,
        )


def simulate_centered_survival(
    n: int = 1200,
    n_features: int = 8,
    n_centers: int = 8,
    base_hazard: float = 0.015,
    censoring_scale: float = 55.0,
    seed: int = 123,
) -> SurvivalData:
    """Simulate survival data with center-specific baseline hazards.

    The covariate effect is shared across centers, but each center has its own
    baseline hazard. This is exactly the setting where a stratified Cox model is
    useful: compare patients within center while learning one shared risk model.
    """

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype("float32")
    center = rng.integers(0, n_centers, size=n, endpoint=False)

    beta = np.zeros(n_features, dtype="float32")
    beta[: min(5, n_features)] = np.array([0.80, -0.65, 0.45, 0.35, -0.30])[: min(5, n_features)]

    center_log_baseline = rng.normal(loc=0.0, scale=0.75, size=n_centers)
    log_hazard = x @ beta + center_log_baseline[center]
    event_rate = base_hazard * np.exp(log_hazard)

    event_time = rng.exponential(scale=1.0 / event_rate)
    censor_time = rng.exponential(scale=censoring_scale, size=n)
    duration = np.minimum(event_time, censor_time).astype("float32")
    event = (event_time <= censor_time).astype("float32")

    feature_names = [f"x{i + 1}" for i in range(n_features)]
    return SurvivalData(
        x=x,
        duration=duration,
        event=event,
        center=center.astype("int64"),
        true_log_risk=(x @ beta).astype("float32"),
        feature_names=feature_names,
    )


def train_test_split(data: SurvivalData, test_size: float = 0.25, seed: int = 123) -> tuple[SurvivalData, SurvivalData]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(data.x.shape[0])
    n_test = int(round(test_size * data.x.shape[0]))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return data.subset(train_idx), data.subset(test_idx)
