---
name: review-training
description: Review baseline stock-model training logs and MLflow results for correctness, reproducibility, and overfitting. Use manually after the Concourse baseline-training job finishes.
disable-model-invocation: true
context: fork
agent: Explore
background: false
disallowed-tools: Write Edit Bash Agent
---

# Review Training

Review the supplied training logs, configuration, and MLflow reports: $ARGUMENTS

1. Verify chronological splits, untouched test data, train-only preprocessing, seeds, dataset version, feature version, and target definition.
2. Compare training and validation curves and report convergence, instability, or overfitting.
3. Compare results with simple baselines and report class imbalance and per-symbol or per-regime weaknesses.
4. Confirm parameters, metrics, artifacts, model file, scaler, environment, and run ID were recorded.
5. Do not edit files, execute commands, retrain models, register models, start jobs, or grant approval.

Return only:

- **Status:** `PASS`, `FAIL`, or `REQUIRES_REVIEW`
- **Evidence:** concise findings with source paths or run IDs
- **Blocking issues:** items that must be resolved
- **Recommendation:** whether regularization tuning is justified
- **Approval:** `Human approval required`