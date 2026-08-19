# Phase 4 Shared Label-Policy Contract

## Frozen policy

- policy: `atr_relative_3class_v1`
- ATR period: 14
- multiplier: 0.75
- horizon: 15m

Do not alter label semantics.

## Requirement

Build or attach one shared label artifact against the Phase-3 common sample universe.

CONTROL and ACD must use exactly the same labels.

Preferred new output root:

`data/parquet/label_policy_phase4_common_v1`

Do not overwrite production label-policy artifacts.

## Pre-training assertions

1. label rows align to the Phase-3 shared universe;
2. sequence keys align;
3. split assignments align;
4. no pre-2020-01-03 samples;
5. CONTROL and ACD resolve to identical labels;
6. policy remains ATR14 × 0.75 with 15m horizon;
7. TEST is not used for fitting or model-selection decisions.

Any adaptation/validation computation required for this handoff must exist in persistent Python code suitable for Concourse invocation.
