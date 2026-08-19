# Causal Integration Contract — Reference

## Fixed constraints
- Symbol: NVDA.
- Common modeling start: `2020-01-03`.
- State-machine outputs are read-only external research artifacts.
- No future information may enter an earlier feature row.
- Preserve current market-ml split/training methodology for the first controlled comparison.
- Re-baselined control and ACD-enhanced datasets must use the same timestamp universe.
- Filter/join first, then rebuild sequences.
- Preserve OR warm-up rows and use `sm_available=0`.
- Do not drop opening 5/10/15-minute rows just because OR is still forming.
- Do not mix all six OR/ATR configurations in the first experiment.
- Prefer normalized geometry over raw absolute A/C price levels.

## Phase-1 representative config
`nvda__or5__atr14__a010__c020__rules-v2`

## Candidate causal families
If actual timing confirms:
- atomic/previous state and state-change indicators
- price-zone context
- OR/ATR availability flags
- normalized distances to OR/A/C levels
- OR width/buffer geometry normalized by ATR
- environment descriptors available after OR finalization
- regime direction/phase/age
- causal confirmation/reinforcement/reentry/opposite-confirmation counters
- causal retest/retrace/ambiguity/strength context
- provenance IDs for signal config, environment schema/threshold set, and regime policy

## Forbidden families
- signal outcomes/scores
- future-direction correctness
- MFE/MAE at resolution
- future-dependent resolution timestamps
- regime evaluation metrics
- end-of-day summaries used for earlier same-day rows

## Redundancy policy
Phase 1 flags exact and near redundancies; it does not automatically remove them. Correlation analysis belongs later, after the feature table exists.
