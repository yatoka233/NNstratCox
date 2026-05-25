from __future__ import annotations

import numpy as np
import torch

from .loss import stratified_cox_loss


def _cindex_one_group(event: np.ndarray, duration: np.ndarray, risk: np.ndarray) -> tuple[float, int]:
    event = np.asarray(event).astype(bool)
    duration = np.asarray(duration, dtype=float)
    risk = np.asarray(risk, dtype=float)

    concordant = 0.0
    comparable = 0
    for i in range(duration.shape[0]):
        if not event[i]:
            continue
        mask = duration[i] < duration
        if not np.any(mask):
            continue
        comparable += int(mask.sum())
        concordant += float(np.sum(risk[i] > risk[mask]))
        concordant += 0.5 * float(np.sum(risk[i] == risk[mask]))

    if comparable == 0:
        return float("nan"), 0
    return concordant / comparable, comparable


def concordance_index(
    event: np.ndarray,
    duration: np.ndarray,
    risk: np.ndarray,
    strata: np.ndarray | None = None,
) -> float:
    """Harrell C-index, optionally restricted to within-stratum comparisons."""

    if strata is None:
        score, _ = _cindex_one_group(event, duration, risk)
        return score

    strata = np.asarray(strata)
    weighted_sum = 0.0
    total_pairs = 0
    for stratum in np.unique(strata):
        idx = strata == stratum
        score, pairs = _cindex_one_group(event[idx], duration[idx], risk[idx])
        if pairs == 0 or np.isnan(score):
            continue
        weighted_sum += score * pairs
        total_pairs += pairs

    if total_pairs == 0:
        return float("nan")
    return weighted_sum / total_pairs


def _encoded_strata(strata: np.ndarray | None, n: int) -> np.ndarray:
    if strata is None:
        return np.zeros(n, dtype=np.int64)

    strata = np.asarray(strata).reshape(-1)
    if strata.shape[0] != n:
        raise ValueError("strata must have the same length as event, duration, and risk")

    _, encoded = np.unique(strata, return_inverse=True)
    return encoded.astype(np.int64)


def _partial_likelihood_deviance(
    event: np.ndarray,
    duration: np.ndarray,
    risk: np.ndarray,
    strata: np.ndarray | None,
) -> float:
    event = np.asarray(event, dtype=np.float64).reshape(-1)
    duration = np.asarray(duration, dtype=np.float64).reshape(-1)
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)

    if not (event.shape[0] == duration.shape[0] == risk.shape[0]):
        raise ValueError("event, duration, and risk must have the same length")

    strata_encoded = _encoded_strata(strata, event.shape[0])
    loss = stratified_cox_loss(
        torch.as_tensor(risk, dtype=torch.float64),
        torch.as_tensor(duration, dtype=torch.float64),
        torch.as_tensor(event, dtype=torch.float64),
        torch.as_tensor(strata_encoded, dtype=torch.long),
    )
    return float((2.0 * loss).detach().cpu().item())


def predictive_deviance(event: np.ndarray, duration: np.ndarray, risk: np.ndarray) -> float:
    """Test Cox partial-likelihood deviance.

    This is ``-2 * log partial likelihood`` divided by the number of observed
    events. Smaller values indicate better held-out partial likelihood.
    """

    return _partial_likelihood_deviance(event, duration, risk, strata=None)


def stratified_predictive_deviance(
    event: np.ndarray,
    duration: np.ndarray,
    risk: np.ndarray,
    strata: np.ndarray,
) -> float:
    """Test stratified Cox partial-likelihood deviance.

    Risk sets are restricted to subjects in the same stratum. The returned value
    is ``-2 * log partial likelihood`` divided by the number of observed events.
    Smaller values indicate better held-out stratified partial likelihood.
    """

    return _partial_likelihood_deviance(event, duration, risk, strata=strata)
