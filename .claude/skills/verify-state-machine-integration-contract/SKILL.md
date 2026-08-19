---
name: verify-state-machine-integration-contract
description: Resolve the blocking causal and integration issues found in the Milestone 1 Phase 1 state-machine integration report, without implementation.
---

# Milestone 1 — Phase 1 Verification

## Purpose
Perform a narrow read-only verification pass on the existing Phase 1 report. Do not rescan the full repository and do not rewrite unaffected sections.

Use the prior Milestone 0 and Phase 1 findings already in context.

## Inspect only what is needed to resolve these blockers

1. Verify ACD-vs-market-ml OHLCV identity on the common NVDA timestamps.
   - Use `state/state_trace_1min.parquet` OHLCV and the authoritative market-ml 1-minute source/feature lineage.
   - Report exact matched rows, unmatched rows, duplicate-key counts, and OHLCV mismatch counts/tolerances.
   - Normalization is NOT an acceptable substitute for source-identity verification.

2. Verify the frozen environment-category causality issue.
   - `category_fit_session_end = 2022-12-30`.
   - Determine whether `or_width_class`, `buffer_width_class`, `spacing_class`, `environment_id`, or any regime/context feature depends on thresholds fitted using future sessions relative to 2020-2022 rows.
   - Inspect ACD engine code/config read-only if required.
   - If a feature inherits this future-fitted information, classify it unsafe before the cutoff unless a causal rebuild is defined.

3. Verify `acd_today_*` and rolling `*_reliability_*` calculations from engine code.
   - Safe only if each row uses information observable at or before that row.
   - If not provable, exclude from the first Phase-2 whitelist.

4. Freeze the environment join rule.
   - No carry-forward from a previous session.
   - Join only within the same `session_date/trading_date`.
   - Environment values are unavailable until that session's `or_finalized_at`.
   - `prediction_time < or_finalized_at` must remain unavailable for that session.

5. Freeze sample-universe invariance.
   - ACD availability must NEVER determine whether a market-ml row is kept.
   - Use a left join from the common market-ml timestamp universe.
   - Availability flags gate feature values only.
   - Baseline-control and ACD-enhanced variants must have identical eligible timestamps and sequences.

6. Resolve the Phase-2 materialization contract.
   - If existing `sequences.py` / `model_matrix.py` are reused unchanged, specify exactly where the combined `core_v1 + ACD` feature table is materialized.
   - Do not leave the consumer path ambiguous.
   - Never overwrite `core_v1`.

7. Regime policy handling.
   - Builder must be parameterized by policy.
   - Do not concatenate policies into one feature set in the first experiment.
   - If one policy is needed for a smoke implementation, use one explicitly named policy and keep the choice configurable.

## Required output
Return only:

A. Blocking issue verdict table: PASS / FAIL / UNRESOLVED  
B. Corrected causal-safe Phase-2 feature whitelist  
C. Corrected forbidden/deferred list  
D. Exact join + availability contract  
E. Exact materialization contract  
F. Any remaining blocker to Phase 2

Keep it concise. No implementation. Stop after the verification report.
