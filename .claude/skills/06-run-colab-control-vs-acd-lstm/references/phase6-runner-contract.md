# Phase-6 Persistent Runner Contract

## Preferred entry point

`scripts/run_phase6_colab_control_vs_acd.py`

Use the existing trusted trainer for actual model construction/training.
Phase 06 should orchestrate, validate, and report; it should not duplicate the
LSTM implementation.

## Required modes

### `--mode preflight`

Non-training. Verify:

- Colab runtime and GPU
- TensorFlow/Keras versions
- Drive/repo paths
- Phase-5 integrity PASS
- generated CONTROL/ACD configs exist
- paired sequence and label identities/counts
- expected CONTROL/ACD feature identities/counts
- output root writable
- TEST evaluation disabled

Write a machine-readable preflight report.

### `--mode train-control`

Explicit human-invoked CONTROL training only.

Before training, rerun/consume the same preflight checks and fail closed on any
mismatch.

### `--mode train-acd`

Explicit human-invoked ACD training only, under the same frozen contract.

### `--mode compare-validation`

Read completed CONTROL and ACD run reports and build one paired VALIDATION-only
comparison. Never call a TEST evaluator.

## No `--mode all`

Do not provide a convenience mode that trains both variants automatically.
Accidental GPU training should be difficult.

## Fail-closed behavior

Training modes must non-zero fail if:

- not in Google Colab;
- no TensorFlow GPU is visible;
- approved dependency versions mismatch;
- Phase-5 report is absent or FAIL;
- shared sequence/label identities mismatch;
- config integrity is no longer valid;
- feature counts differ from the approved experiment;
- output run directory already represents a completed run and overwrite was not
  explicitly authorized;
- any TEST evaluation option is requested.

## Local/static testability

Structure imports so local py_compile/unit tests do not need TensorFlow merely
to parse the runner. Heavy training imports should occur only inside the Colab
runtime path that needs them.
