# Deep Stratified Cox Tutorial

This directory is a small, standalone tutorial for fitting a deep stratified Cox
model. It does not use KL distillation, teacher models, or teacher-student
training. By default it uses full same-stratum risk sets.

The model is:

```text
neural network risk score f_theta(X)
+ stratified Cox partial likelihood
```

The stratum variable can be a hospital center, transplant center, site, matched
group, or any variable that defines separate baseline hazards. The model learns a
shared covariate-risk function while comparing subjects only within the same
stratum.

## Files

```text
NNstratCox/
  README.md
  tutorial_deep_stratified_cox.py
  nnstratcox/
    model.py       # MLP risk model
    dataset.py     # event-centered same-stratum risk-set batches
    loss.py        # Breslow Cox loss for full strata and components
    train.py       # batched event-centered risk-set training loop
    metrics.py     # ordinary and stratified C-index
    simulate.py    # simulated center-stratified survival data
```

## Run The Tutorial

Use any Python environment with PyTorch and NumPy installed. From this
directory:

```bash
cd /path/to/NNstratCox
pip install -r requirements.txt
```

Run the default simulated-data tutorial:

```bash
python -u tutorial_deep_stratified_cox.py
```

Run a short smoke test first:

```bash
python -u tutorial_deep_stratified_cox.py --n 80 --epochs 3 --batch-size 4 --device cpu
```

Run a larger CPU example:

```bash
python -u tutorial_deep_stratified_cox.py --n 600 --n-centers 8 --epochs 100 --batch-size 16 --device cpu
```

Use `--device cuda` if the active environment has a GPU-enabled PyTorch build.

Important command-line arguments:

```text
--n            number of simulated subjects
--n-centers    number of simulated hospital centers / strata
--epochs       maximum training epochs
--batch-size   number of event-centered risk-set components per optimizer step
--device       cpu, cuda, or unset for automatic selection
```

## Data Format

For your own data, prepare four arrays:

```python
X          # shape (n, p), numeric covariates
duration   # shape (n,), observed event/censoring times
event      # shape (n,), 1 for event, 0 for censored
strata     # shape (n,), integer-coded hospital/center labels
```

The stratum labels define separate baseline hazards. If the raw center labels
are strings, first encode them as integer codes.

## Stratified Cox Target

The neural net outputs one scalar risk score:

```text
h_i = f_theta(X_i)
```

For an event subject `i`, the stratified Cox denominator only includes subjects
from the same stratum:

```text
R_i = {j: duration_j >= duration_i and strata_j == strata_i}
loss_i = -h_i + log sum_{j in R_i} exp(h_j)
```

So if `strata` is hospital center, every event is compared only against patients
at risk in the same hospital center. The model still shares the same neural
network parameters across all centers; only the Cox risk-set denominators are
stratified.

The full stratified Cox loss sums this over observed events and divides by the
number of events. Tied event times are handled with the Breslow approximation in
`nnstratcox/loss.py`.

## Minimal Usage

```python
from nnstratcox import MLPRisk, fit_model, predict_risk, standardize_train_test

x_train, x_test = standardize_train_test(x_train, x_test)

model = MLPRisk(input_dim=x_train.shape[1], hidden_dim=64, num_layers=2)

history = fit_model(
    model,
    x_train,
    duration_train,
    event_train,
    center_train,
    x_val=x_test,
    duration_val=duration_test,
    event_val=event_test,
    strata_val=center_test,
    batch_size=32,
)

risk_score = predict_risk(model, x_test)
```

## How Batching Works

This tutorial does not use ordinary random patient mini-batches. Instead, it
uses batches of event-centered Cox components, mirroring the stratum-batch style
used in the larger project.

`StratifiedRiskSetDataset` builds one dataset item per observed event case. For
event case `i`, one item contains:

```text
[event case i] + [all patients j with duration_j >= duration_i and strata_j == strata_i]
```

The local event label is rebuilt inside the component:

```text
event_i = 1 for the event case
event_j = 0 for every other subject in this component
```

This means censored subjects and later-failing subjects can appear as controls
in another event case's risk set. They are not treated as events for that local
component.

A `DataLoader` batch contains several variable-length components. The custom
`risk_set_collate` function returns lists instead of stacking them, because each
event case can have a different number of at-risk subjects. In each optimizer
step, `fit_model` computes:

```text
batch_loss = mean(component_loss_1, ..., component_loss_B)
```

where `B` is `--batch-size`, the number of event-centered risk sets in the
batch.

With the default `max_controls=None`, every component uses the full same-stratum
risk set, so this is still the exact stratified Cox target, optimized in
component batches. If `max_controls` is set, controls are sampled and the method
becomes a sampled risk-set / NCC-style approximation.

## Loss Implementation

The loss code has two entry points:

```text
stratified_cox_loss(log_risk, duration, event, strata)
cox_component_loss(log_risk, duration, event)
```

`stratified_cox_loss` is the full-data evaluation loss. It loops over unique
strata, computes the Cox negative partial log-likelihood inside each stratum,
sums the stratum losses, and divides by the total number of events.

`cox_component_loss` is used during training. Each component already contains
one same-stratum risk set, so it does not need a `strata` argument. It computes
the Cox loss on that event-centered component.

Inside `_cox_breslow_nll_for_one_stratum`, subjects are sorted by observed time.
The denominator for each event time is computed by a reverse cumulative
log-sum-exp:

```text
log_risk_set_sum[k] = log sum_{m: duration_m >= duration_k} exp(h_m)
```

For tied times, events with the same observed time are grouped. The Breslow
negative log-likelihood contribution for one time group is:

```text
-(sum event scores at this time
  - number of events at this time * log risk-set sum at this time)
```

The returned loss is normalized by the number of observed events, so loss values
are comparable across batches and validation sets with different event counts.

## What This Is Not

This tutorial does not implement:

- teacher-student KL distillation
- component-wise KL loss
- OPTN-specific data processing

It is a clean teaching version of deep stratified Cox only.
