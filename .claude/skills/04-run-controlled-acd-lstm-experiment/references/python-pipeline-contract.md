# Python Pipeline Contract

## Goal

Important pipeline logic must be reusable without Claude and callable later from Concourse CI.

## Persistent Python requirement

Any computation required to regenerate or validate project artifacts must exist in repository Python code.

Examples:

- feature generation;
- state-machine feature generation;
- categorical encoding;
- sequence generation;
- model-matrix creation;
- label-policy creation;
- training;
- evaluation;
- comparison;
- reproducibility validation.

Preferred locations:

- `src/*.py`
- `scripts/*.py`
- `tests/*.py`

## Phase-4 entry point

Required:

`scripts/run_phase4_control_vs_acd.py`

Recommended modes:

- `--mode labels`
- `--mode preflight`
- `--mode train-control`
- `--mode train-acd`
- `--mode compare`
- `--mode validate`

## Design rules

The Python entry point should:

1. use existing project modules rather than duplicate algorithms;
2. resolve paths programmatically;
3. read project configuration where appropriate;
4. produce deterministic output identities;
5. fail with non-zero exit code when a contract is violated;
6. print clear PASS/FAIL;
7. be callable by a human or Concourse CI;
8. avoid hidden session state;
9. avoid manual feature lists when they can be derived from manifests/config;
10. avoid dependence on Claude.

## Bash/shell rule

Do not implement core pipeline behavior in Bash.

If shell is necessary for CI task wiring, it should only invoke persistent Python entry points.
