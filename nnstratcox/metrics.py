from __future__ import annotations

import numpy as np


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
