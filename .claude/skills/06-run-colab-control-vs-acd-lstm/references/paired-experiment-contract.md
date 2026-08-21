# Paired Experiment Contract

## Objective

Measure the incremental value of the approved ACD/state-machine feature set
without changing any other part of the experiment.

## Frozen common universe

Current approved Phase-3/Phase-4 counts:

- total sequences: 1,259,396
- TRAIN: 878,828
- VALIDATION: 189,428
- TEST: 191,140

TEST is sealed in Phase 06.

## Variants

CONTROL:

- feature_set: `core_v1`
- feature count: 23

ACD:

- feature_set: `core_v1_acd_v1`
- feature count: 150

The CONTROL 23 features must be a subset of ACD's 150 features.

## Shared labels

- policy: `atr_relative_3class_v1`
- ATR period: 14
- horizon: 15m
- multiplier: 0.75
- approved current signature prefix: `cf167954f3b2`

Phase 06 must consume the shared label artifact. Do not create separate CONTROL
and ACD labels.

## Shared settings

The runner must load and compare the generated CONTROL/ACD configs and fail if
any unapproved non-feature difference appears.

The following must be equal across variants:

- sequence length/stride/scope/sessions
- split assignment/purge rules
- scaler policy
- label policy
- target horizon
- model architecture
- optimizer/loss
- learning rate and weight decay
- batch size/epochs
- early stopping
- gradient clipping
- seed
- mixed precision
- worker/data-loader settings that affect numerics
- class weights or sampling behavior

Do not separately tune either variant.

## Experiment revision rule

If runtime limitations require a material change to batch size, precision,
architecture, optimizer, label policy, or another frozen field, stop.

Do not silently change the setting for one or both variants. Create a reviewed
new experiment revision and rerun both variants under the same new contract.
