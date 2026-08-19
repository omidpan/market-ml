# Phase 3 Experiment Contract

## Goal

Isolate the incremental effect of ACD/state-machine features.

The paired experiment is valid only if CONTROL and ACD differ in feature columns, not in samples.

## CONTROL

Source:
`data/parquet/features_1m`

Common start:
`2020-01-03`

Use the existing approved baseline feature set only.

Rebuild sequences/model matrix from the common universe. Do not reuse older sequence artifacts created from a different timestamp range.

## ACD

Source:
`data/parquet/features_1m_acd_v1`

Common start:
`2020-01-03`

Use the same baseline columns plus the approved ACD feature whitelist.

## Must remain identical between variants

- symbol
- timestamp universe
- session inclusion
- label policy
- horizon
- target rows
- split labels
- sequence length
- stride
- purge/embargo
- sequence start/end timestamps
- prediction_time
- target_realized_at
- sample ordering

## First experiment only

Use:
- OR = 5
- ATR = 14
- A = 0.10 ATR
- C = 0.20 ATR
- rules v2
- regime policy = moderate_or_anchored-r1

Do not add consensus across policies.
Do not add other OR/ATR configs.
