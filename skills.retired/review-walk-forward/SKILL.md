---
name: review-walk-forward
description: Review walk-forward stock-model validation for leakage, stability, and correct time-series evaluation. Use manually after the Concourse walk-forward job finishes.
disable-model-invocation: false
context: fork
agent: Explore
background: false
disallowed-tools: Write Edit Bash Agent
---

# Review Walk-Forward Validation

Review the supplied fold definitions, logs, predictions, and metric reports: $ARGUMENTS

1. Verify chronological folds, retraining boundaries, purge or embargo requirements, and train-only preprocessing in every fold.
2. Confirm predictions are strictly out of sample and no fold uses future labels, features, scalers, or tuning information.
3. Review fold dispersion, degradation over time, symbol and regime consistency, class balance, and confidence calibration.
4. Flag conclusions based only on an average metric when individual folds are unstable.
5. Do not edit files, execute commands, rerun validation, start jobs, or grant approval.

Return only:

- **Status:** `PASS`, `FAIL`, or `REQUIRES_REVIEW`
- **Evidence:** concise fold-level findings with source paths
- **Stability assessment:** robust, mixed, or unstable
- **Blocking issues:** items that must be resolved
- **Recommendation:** whether the user should consider cost-aware backtesting
- **Approval:** `Human approval required`