# Colab Execution Contract

## Human invocation only

Claude Code prepares the code and runbook. The user starts the actual Colab GPU
runtime and manually invokes Phase-6 commands.

No local workstation, Claude execution environment, or Concourse worker may run
full LSTM training.

## Thin Colab bootstrap

A notebook is optional. If created, it should contain only simple cells for:

1. selecting/checking GPU runtime;
2. mounting Google Drive;
3. changing to the repository directory;
4. installing `requirements-train.txt`;
5. invoking the persistent Phase-6 runner.

Do not duplicate training code in notebook cells.

## Recommended execution order

1. preflight
2. inspect PASS report
3. train CONTROL
4. verify CONTROL artifacts are complete
5. train ACD
6. verify ACD artifacts are complete
7. compare VALIDATION

If a Colab session disconnects, do not infer success from partial files. Each
run must have an explicit completed status/manifest before comparison.

## Separate-session allowance

CONTROL and ACD may run in separate Colab sessions if needed. Each run must
record runtime/dependency/config identity so the comparison can prove they were
run under the same approved experiment contract.
