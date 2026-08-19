---
name: build-state-machine-features
description: Implement the approved Milestone 1 Phase 2 causal state-machine feature layer for market-ml without changing the existing training logic.
---

# Milestone 1 — Phase 2: Build State-Machine Feature Layer

## Purpose

Implement the approved, leakage-safe ACD/state-machine feature layer for `market-ml`.

This phase must preserve:
- `data/raw/` as immutable ground truth;
- existing `core_v1`;
- current label logic;
- current chronological split methodology;
- current sequence and model-matrix semantics;
- point-in-time causality.

Do not run full model training in this phase.

## Required inputs

Use the approved Milestone 1 Phase 1 integration contract and verification results.

Representative configuration for the first implementation:

`nvda__or5__atr14__a010__c020__rules-v2`

State-machine root:

`$HOME/acd_experiments_local/nvda/`

Regime policy must be explicit and configurable. Do not concatenate multiple policies into one feature set.

## Implementation scope

Create the smallest production-quality feature integration path that:

1. reads only approved causal ACD sources;
2. validates source identity and keys before joining;
3. uses the market-ml timestamp universe as the left/base universe;
4. preserves opening-range rows with availability flags;
5. materializes a versioned state-machine feature layer;
6. materializes a combined feature-set identity without overwriting `core_v1`;
7. allows existing sequence/model-matrix builders to consume the new feature set;
8. records provenance for signal config, regime policy, environment schema, threshold set, and feature schema;
9. never uses ACD availability to drop market-ml rows.

## Hard exclusions

Do not read or use as features:
- `outcomes/signal_outcomes.parquet`
- `regimes/*/regime_evaluation.parquet`
- future returns
- future direction correctness
- MFE / MAE outcome data
- end-of-day summaries for earlier intraday timestamps
- any unresolved or unverified same-day / rolling reliability field
- any frozen category or regime field that Phase 1 verification classified as future-fitted or unsafe

## Join contract

Follow the approved contract in `references/implementation-contract.md`.

Key rule:

`market-ml common universe -> left join ACD -> availability masking -> feature construction -> combined feature set -> sequence rebuild`

Never make ACD availability part of row eligibility.

## Files

Preferred implementation shape:

- new `src/state_machine_features.py`
- minimal `config/pipeline.yaml` addition for a new feature-set identity
- tests under `tests/`
- no behavioral edits to `src/train_model.py`
- avoid editing `src/features.py`, `src/sequences.py`, or `src/model_matrix.py` unless a small interface change is strictly required and justified

If a smaller compatible design is possible, prefer it.

## Phase workflow

### Step 1 — Plan
Before editing:
- inspect only the files necessary to implement this phase;
- identify exact files to create/modify;
- identify exact output paths and schema;
- identify exact tests;
- identify any conflict with the approved Phase 1 contract.

Stop and wait for the exact approval phrase:

`Approved—implement it`

### Step 2 — Implement
After approval:
- implement only the approved scope;
- do not broaden the task;
- do not touch unrelated files;
- do not run full training.

### Step 3 — Validate
Run only targeted validation described in `references/test-contract.md`.

## Required completion report

Return:
1. files created/modified;
2. final feature schema;
3. output/materialization paths;
4. provenance fields;
5. join/availability behavior;
6. tests run and results;
7. row-count invariance results;
8. source-identity validation results;
9. any deferred fields;
10. exact command for the first dry-run build;
11. exact command for building the re-baselined control;
12. exact command for building the ACD-enhanced variant;
13. anything still blocking Phase 3 experiment preparation.

Do not start model training.
