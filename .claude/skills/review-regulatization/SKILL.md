---
name: review-regularization
description: Review stock-model regularization and tuning results against an approved baseline. Use manually after the Concourse regularization-tuning job finishes.
disable-model-invocation: true
context: fork
agent: Explore
background: false
disallowed-tools: Write Edit Bash Agent
---

# Review Regularization

Review the supplied tuning logs, baseline comparison, and MLflow reports: $ARGUMENTS

1. Review L1/L2 penalties, dropout, weight decay, early stopping, learning-rate controls, and model complexity actually tested.
2. Confirm tuning used training and validation periods only and did not inspect or optimize against the final test period.
3. Compare every candidate with the same baseline, dataset, split, features, seed policy, and metrics.
4. Reject improvements that are negligible, unstable, or limited to one symbol, horizon, or regime.
5. Do not edit files, execute commands, retrain models, choose a production model, start jobs, or grant approval.

Return only:

- **Status:** `PASS`, `FAIL`, or `REQUIRES_REVIEW`
- **Evidence:** concise comparison with run IDs
- **Overfitting assessment:** key observations
- **Blocking issues:** items that must be resolved
- **Recommendation:** candidate configuration for human consideration
- **Approval:** `Human approval required`