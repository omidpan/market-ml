---
name: design-state-machine-feature-integration
description: Define the causal, leakage-safe integration contract between market-ml and ACD state-machine outputs before implementation or training.
---

# Milestone 1 — Phase 1: State-Machine Feature Integration Contract

## Purpose
Define the exact causal integration contract between `market-ml` and ACD state-machine outputs before feature implementation, correlation analysis, or model training.

This phase is read-only. Do not modify repository files, Parquet data, model artifacts, Docker configuration, or training code.

## Token-efficiency rule
Use Milestone 0 understanding already established. Do not rescan the full repository.

Inspect only what is needed from current authoritative market-ml feature/target/sequence/model-matrix code and one representative state-machine configuration:

`$HOME/acd_experiments_local/nvda/nvda__or5__atr14__a010__c020__rules-v2`

Inspect actual schemas for:
- `state/state_trace_1min.parquet`
- `context/context_1min.parquet`
- `environment/environment_daily.parquet`
- regime-context outputs for the 3 policies
- relevant manifests/config metadata

## Hard leakage exclusions
Never propose as model inputs:
- `signal_outcomes.parquet`
- `regime_evaluation.parquet`
- future returns or future-direction correctness
- MFE/MAE outcome fields
- resolution outcomes
- end-of-day same-session summaries used for earlier timestamps
- any value depending on future bars relative to the feature timestamp

If timing is uncertain, classify as unresolved rather than safe.

## Required decisions
Define:
1. exact timestamp/session join keys;
2. common NVDA modeling start = `2020-01-03`;
3. OR warm-up handling with `sm_available=0` rather than dropping rows;
4. causally safe state-machine fields;
5. forbidden/leakage fields;
6. raw/redundant fields not suitable as default ML features;
7. normalized candidate features from atomic state, price zone, A/C geometry, environment, and regime context;
8. categorical encoding requirements;
9. missing-value behavior;
10. availability timestamp for every candidate;
11. provenance columns for ACD config/policy;
12. rule that filtering/joining happens before sequence rebuilding;
13. re-baselined control dataset with the identical timestamp universe but no ACD features.

## Experimental boundary
Do not:
- compare all six OR/ATR configs yet;
- concatenate ACD configs;
- run correlation analysis;
- train a model;
- write Parquet;
- change the training pipeline or split methodology;
- implement `statemachine_features.py`.

## Required report
Return only:

### A. Source-table/schema map
Path, grain, join keys, important columns, availability timing, provenance.

### B. Causal-safe feature candidate table
Feature name, source columns, transformation, type, availability timestamp, missing/warm-up behavior, redundancy note, evidence status.

### C. Forbidden/leakage feature table
Source, reason, leakage mechanism, permanently forbidden vs only unsafe intraday.

### D. Timestamp/join contract
Timezone, timestamp convention, exact keys, duplicate handling, cardinality, common start filter, daily-to-intraday environment join.

### E. Missing/warm-up contract
`sm_available`, OR warm-up, invalid ATR/environment, regime-not-yet-established behavior.

### F. Sequence reconstruction contract
Required order:
`filter common universe -> exact join -> feature construction -> session validity checks -> sequence rebuild`

Do not reuse old sequences if their timestamp universe differs.

### G. Proposed Phase-2 implementation files
List only. Do not create.

### H. Unresolved questions
Only genuine design decisions or missing evidence.

## Evidence labels
Use only:
`CONFIRMED_FROM_CODE`, `CONFIRMED_FROM_ARTIFACT`, `CONFIRMED_FROM_TEST`, `DOCUMENTED_ONLY`, `INFERRED`, `UNKNOWN`.

Keep the report concise and evidence-based. Stop after the report and wait for explicit approval.
