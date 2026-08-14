---
name: review-feature-engineering
description: Review stock feature-engineering logs, metadata, and validation reports for leakage and point-in-time correctness. Use manually after the Concourse feature jobs finish.
disable-model-invocation: true
context: fork
agent: Explore
background: false
disallowed-tools: Write Edit Bash Agent
---

# Review Feature Engineering

Review the supplied log, metadata, or report paths: $ARGUMENTS

1. Check feature formulas, lookback windows, warm-up rows, timestamp alignment, and availability at prediction time.
2. Detect future-bar use, centered windows, backward joins, cross-symbol leakage, and preprocessing fitted outside training data.
3. Review NaN/Inf handling, missing-bar policy, scaling, distributions, drift, constant features, and extreme outliers.
4. Confirm feature names, versions, parameters, source-data version, and generated row counts are recorded.
5. Do not edit files, execute commands, start jobs, or grant approval.

Return only:

- **Status:** `PASS`, `FAIL`, or `REQUIRES_REVIEW`
- **Evidence:** concise findings with source paths
- **Leakage risks:** confirmed and suspected risks
- **Blocking issues:** items that must be resolved
- **Recommendation:** whether the user should consider baseline training
- **Approval:** `Human approval required`