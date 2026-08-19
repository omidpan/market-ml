# Phase 1 Review Constraints

## Strict point-in-time rule
The project rule is stronger than merely preventing validation/test leakage: no future information may be used to define an earlier feature row.

Therefore a category threshold fitted through 2022-12-30 is not automatically causal for 2020-2022 rows merely because the cutoff lies inside the eventual training split.

## Sample-universe rule
State-machine unavailability is a feature state, not a row-eligibility rule.

Required order:

`market-ml common timestamp universe -> left join ACD -> availability masking -> feature construction -> identical sequence eligibility for control and ACD variants`

## Source-identity rule
Because ACD state transitions are functions of its OHLCV input, bar-source equality must be verified directly. ATR-normalized features do not eliminate this dependency.

## Safe-first rule
Phase 2 may proceed with a smaller verified-safe feature whitelist. Unresolved same-day counters, rolling reliability features, frozen categories, or regime fields should be deferred rather than guessed safe.
