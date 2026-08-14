---
name: review-backtest
description: Review quantitative stock backtest logs and reports for realism, leakage, costs, and risk. Use manually after the Concourse backtest job finishes.
disable-model-invocation: true
context: fork
agent: Explore
background: false
disallowed-tools: Write Edit Bash Agent
---

# Review Backtest

Review the supplied backtest configuration, logs, trades, and reports: $ARGUMENTS

1. Verify signals use information available before execution and that execution timing is realistic.
2. Check commissions, bid-ask spread, slippage, latency, turnover, liquidity, position limits, and market-session rules.
3. Review return, volatility, drawdown, Sharpe or Sortino, hit rate, exposure, turnover, and performance by symbol and regime.
4. Check benchmark comparison, sensitivity analysis, multiple-testing risk, survivorship bias, and concentration.
5. Do not edit files, execute commands, rerun backtests, register or deploy models, start jobs, or grant approval.

Return only:

- **Status:** `PASS`, `FAIL`, or `REQUIRES_REVIEW`
- **Evidence:** concise findings with source paths or run IDs
- **Realism assessment:** key assumptions and weaknesses
- **Blocking issues:** items that must be resolved
- **Recommendation:** whether the candidate merits human review for registration
- **Approval:** `Human approval required`