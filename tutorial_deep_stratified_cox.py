from __future__ import annotations

import argparse

import numpy as np
import torch

from nnstratcox import (
    MLPRisk,
    concordance_index,
    fit_model,
    predict_risk,
    simulate_centered_survival,
    standardize_train_test,
    train_test_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep stratified Cox tutorial")
    parser.add_argument("--n", type=int, default=600, help="Number of simulated subjects")
    parser.add_argument("--n-centers", type=int, default=8, help="Number of hospital centers")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of event-centered risk sets per batch")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None, help="cpu, cuda, or leave unset for auto")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = simulate_centered_survival(n=args.n, n_centers=args.n_centers, seed=args.seed)
    train, test = train_test_split(data, test_size=0.25, seed=args.seed + 1)
    x_train, x_test = standardize_train_test(train.x, test.x)

    print("Data summary")
    print(f"  train n={x_train.shape[0]}, test n={x_test.shape[0]}")
    print(f"  train event rate={train.event.mean():.3f}, test event rate={test.event.mean():.3f}")
    print(f"  centers={np.unique(data.center).size}")
    print()

    print("Training a deep stratified Cox model")
    print(f"  batch size = {args.batch_size} event-centered risk sets")
    strat_model = MLPRisk(input_dim=x_train.shape[1], hidden_dim=64, num_layers=2, dropout=0.10, seed=args.seed)
    fit_model(
        strat_model,
        x_train,
        train.duration,
        train.event,
        train.center,
        x_val=x_test,
        duration_val=test.duration,
        event_val=test.event,
        strata_val=test.center,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=args.epochs,
        patience=40,
        batch_size=args.batch_size,
        device=args.device,
        verbose=True,
    )
    risk_strat = predict_risk(strat_model, x_test, device=args.device)

    print()
    print("Training an ordinary Cox neural net baseline")
    ordinary_model = MLPRisk(input_dim=x_train.shape[1], hidden_dim=64, num_layers=2, dropout=0.10, seed=args.seed + 10)
    fit_model(
        ordinary_model,
        x_train,
        train.duration,
        train.event,
        np.zeros_like(train.center),
        x_val=x_test,
        duration_val=test.duration,
        event_val=test.event,
        strata_val=np.zeros_like(test.center),
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=args.epochs,
        patience=40,
        batch_size=args.batch_size,
        device=args.device,
        verbose=False,
    )
    risk_ordinary = predict_risk(ordinary_model, x_test, device=args.device)

    print()
    print("Held-out performance")
    print("  Stratified model")
    print(f"    ordinary C-index:   {concordance_index(test.event, test.duration, risk_strat):.3f}")
    print(f"    stratified C-index: {concordance_index(test.event, test.duration, risk_strat, test.center):.3f}")
    print("  Ordinary Cox baseline")
    print(f"    ordinary C-index:   {concordance_index(test.event, test.duration, risk_ordinary):.3f}")
    print(f"    stratified C-index: {concordance_index(test.event, test.duration, risk_ordinary, test.center):.3f}")

    print()
    print("Use this pattern with your data:")
    print("  X: numeric covariate matrix")
    print("  duration: observed follow-up time")
    print("  event: 1 if event occurred, 0 if censored")
    print("  strata: hospital/center labels used to define within-center risk sets")


if __name__ == "__main__":
    main()
