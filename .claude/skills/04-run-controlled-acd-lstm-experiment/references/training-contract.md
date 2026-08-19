# Phase 4 Frozen Training Contract

Use the current active LSTM baseline only.

Reconfirm exact active values from repository code before implementation.

Expected baseline from prior discovery:

- sequence length: 120
- horizon: 15m
- LSTM: 64 units × 2
- dropout: 0.20
- optimizer: AdamW
- learning rate: 0.001
- weight decay: 0.0001
- clipnorm: 1.0
- seed: 42
- early stopping monitor: validation loss
- patience: 3
- min_delta: 1e-4

If current code differs, report the difference before training.

## CONTROL

- feature set: `core_v1`
- feature count: 23

## ACD

- feature set: `core_v1_acd_v1`
- feature count: 150
- incremental ACD features: 127

Both use:

`data/parquet/model_matrix_phase3_common_v1`

Do not tune either variant independently.

Do not overwrite previous training runs.
