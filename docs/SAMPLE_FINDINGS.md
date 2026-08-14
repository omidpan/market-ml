# Partial AVGO sample findings

The previously inspected partial AVGO sample covered approximately:

```text
2019-05-06 04:00
through
2019-05-22 19:59
```

The important structural observation was that the missing records in that
sample were concentrated at the **start of premarket sessions**, rather than
being random holes throughout regular trading.

Examples included trading dates whose first observed candle was much later
than 04:00, such as approximately:

```text
07:11
07:33
07:30
05:27
06:01
07:32
```

Once observations began, the inspected sample showed much stronger continuity
through the rest of the session.

This is the main reason v1.1 does **not** automatically fill session-leading
premarket gaps.

Carrying the previous day's final price into several hours of missing
premarket data would be causal, but it would still fabricate market behavior
that the datasource did not actually observe.

The new validator therefore reports quality by session:

```text
premarket
regular
aftermarket
```

instead of only giving one daily missing count.

Overnight is not expected for the current historical source and therefore is
not counted as missing.
