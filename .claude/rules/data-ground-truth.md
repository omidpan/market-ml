# Data Ground Truth and Causality Rule

## Raw data

`data/raw/` is immutable ground truth.

Never modify it.

Read-only inspection and validation are allowed.

All derived outputs must be written outside `data/raw/`.

## External ACD data

```text
$HOME/acd_experiments_local/<symbol>/
```

is read-only external research input.

Do not modify or regenerate it from `market-ml` unless the user explicitly changes project scope.

## Causal features

For model decision time `t`:

```text
feature_available_at <= t
```

No future bars, future aggregates, centered windows, target leakage, state-machine outcome leakage, or preprocessing fitted on validation/test.

## Colab experiment tracking

Colab training must remain independent of local MLflow availability.

A successful training run may emit a tracker-ready JSON payload to its versioned run-artifact directory.

The payload must not contain credentials or secret environment values.

Do not make direct local-network tracker connectivity a requirement for training completion.
