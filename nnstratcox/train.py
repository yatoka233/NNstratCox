from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import StratifiedRiskSetDataset, risk_set_collate
from .loss import cox_component_loss, stratified_cox_loss


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x_train - mean) / std).astype("float32"), ((x_test - mean) / std).astype("float32")


def _to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def fit_model(
    model: torch.nn.Module,
    x_train: np.ndarray,
    duration_train: np.ndarray,
    event_train: np.ndarray,
    strata_train: np.ndarray,
    x_val: np.ndarray | None = None,
    duration_val: np.ndarray | None = None,
    event_val: np.ndarray | None = None,
    strata_val: np.ndarray | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 40,
    batch_size: int = 32,
    max_controls: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Train with batches of event-centered same-stratum risk sets.

    ``batch_size`` counts Cox likelihood components, not patients. With
    ``max_controls=None`` every component uses the full same-stratum risk set,
    so the stochastic batching changes optimization but not the likelihood
    target. Setting ``max_controls`` samples controls and turns the objective
    into a sampled risk-set approximation.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)

    has_val = x_val is not None and duration_val is not None and event_val is not None and strata_val is not None
    if has_val:
        x_va = _to_tensor(x_val, device, torch.float32)
        t_va = _to_tensor(duration_val, device, torch.float32)
        e_va = _to_tensor(event_val, device, torch.float32)
        s_va = _to_tensor(strata_val, device, torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = StratifiedRiskSetDataset(
        x_train,
        duration_train,
        event_train,
        strata_train,
        max_controls=max_controls,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=risk_set_collate,
    )
    history = TrainingHistory(train_loss=[], val_loss=[], best_epoch=0)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for xs_list, duration_list, event_list in loader:
            optimizer.zero_grad()
            component_losses = []
            for x_b, t_b, e_b in zip(xs_list, duration_list, event_list):
                x_b = x_b.to(device)
                t_b = t_b.to(device)
                e_b = e_b.to(device)
                component_losses.append(cox_component_loss(model(x_b), t_b, e_b))
            loss = torch.stack(component_losses).mean()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses))
        history.train_loss.append(train_loss)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss_t = stratified_cox_loss(model(x_va), t_va, e_va, s_va)
            val_loss = float(val_loss_t.detach().cpu())
        else:
            val_loss = train_loss
        history.val_loss.append(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(f"epoch={epoch + 1:04d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if stale_epochs >= patience:
            if verbose:
                print(f"early stopping at epoch={epoch + 1}; best_epoch={history.best_epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_risk(model: torch.nn.Module, x: np.ndarray, device: str | torch.device | None = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        risk = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return risk.detach().cpu().numpy()
