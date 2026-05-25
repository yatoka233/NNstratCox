from __future__ import annotations

import torch


def _cox_breslow_nll_for_one_stratum(
    log_risk: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Breslow negative partial log-likelihood for one stratum.

    All tensors are one-dimensional and contain only subjects from one stratum.
    The risk set for an event at time t is all subjects in the same stratum with
    observed time >= t.
    """

    order = torch.argsort(duration)
    t = duration[order]
    h = log_risk[order]
    e = event[order].to(dtype=h.dtype)

    if e.sum() == 0:
        zero = h.sum() * 0.0
        return zero, zero

    log_risk_set_sum = torch.logcumsumexp(h.flip(0), dim=0).flip(0)

    _, counts = torch.unique_consecutive(t, return_counts=True)
    group_starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]]).long()
    group_ids = torch.repeat_interleave(torch.arange(counts.numel(), device=t.device), counts)

    event_score_sum = torch.zeros(counts.numel(), dtype=h.dtype, device=h.device)
    event_count = torch.zeros(counts.numel(), dtype=h.dtype, device=h.device)
    event_score_sum.scatter_add_(0, group_ids, e * h)
    event_count.scatter_add_(0, group_ids, e)

    nll = -(event_score_sum - event_count * log_risk_set_sum[group_starts]).sum()
    return nll, event_count.sum()


def stratified_cox_loss(
    log_risk: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
    strata: torch.Tensor,
) -> torch.Tensor:
    """Mean Breslow Cox partial likelihood loss stratified by `strata`.

    Parameters
    ----------
    log_risk:
        Model output, shape ``(n,)``. Larger values mean higher risk.
    duration:
        Observed follow-up times, shape ``(n,)``.
    event:
        Event indicators, shape ``(n,)``. Use 1 for event and 0 for censored.
    strata:
        Stratum labels, shape ``(n,)``. Examples: hospital id, center id, or a
        matched-set label. The baseline hazard is allowed to differ by stratum,
        and the Cox denominator is computed only within the same stratum.

    Returns
    -------
    torch.Tensor
        Negative partial log-likelihood divided by the number of events.
    """

    log_risk = log_risk.reshape(-1)
    duration = duration.reshape(-1)
    event = event.reshape(-1).float()
    strata = strata.reshape(-1)

    if not (log_risk.numel() == duration.numel() == event.numel() == strata.numel()):
        raise ValueError("log_risk, duration, event, and strata must have the same length")

    total_nll = log_risk.sum() * 0.0
    total_events = log_risk.new_tensor(0.0)

    for stratum in torch.unique(strata):
        idx = strata == stratum
        nll_s, events_s = _cox_breslow_nll_for_one_stratum(
            log_risk[idx],
            duration[idx],
            event[idx],
        )
        total_nll = total_nll + nll_s
        total_events = total_events + events_s

    if total_events.item() == 0:
        raise ValueError("Cannot compute Cox loss: no observed events in this batch")

    return total_nll / total_events


def cox_component_loss(
    log_risk: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Cox loss for one event-centered component.

    This is used by the batched trainer. Each component already contains one
    same-stratum risk set, so no stratum labels are needed inside the component.
    """

    nll, n_events = _cox_breslow_nll_for_one_stratum(
        log_risk.reshape(-1),
        duration.reshape(-1),
        event.reshape(-1),
    )
    if n_events.item() == 0:
        raise ValueError("Cannot compute component loss: component has no event")
    return nll / n_events
