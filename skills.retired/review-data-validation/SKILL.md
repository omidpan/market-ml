---
name: review-data-validation
description: Review quantitative market-data validation logs and reports before feature engineering. Use manually after the Concourse data-validation job finishes.
disable-model-invocation: true
context: fork
agent: Explore
background: false
disallowed-tools: Write Edit Bash Agent
---

# Review Data Validation

Review the supplied log or report paths: $ARGUMENTS

1. Verify schema, row counts, symbols, timestamp order, uniqueness, timezone, sessions, OHLC consistency, volume validity, missing bars, gaps, and NaN/Inf counts.
2. Distinguish expected market closures from unexplained gaps.
3. Confirm raw data was not modified and derived outputs are versioned.
4. Identify evidence of corruption, survivorship bias, or look-ahead contamination.
5. Do not edit files, execute commands, start jobs, or grant approval.

Return only:

- **Status:** `PASS`, `FAIL`, or `REQUIRES_REVIEW`
- **Evidence:** concise findings with source paths
- **Blocking issues:** items that must be resolved
- **Recommendation:** whether the user should consider feature engineering
- **Approval:** `Human approval required`
