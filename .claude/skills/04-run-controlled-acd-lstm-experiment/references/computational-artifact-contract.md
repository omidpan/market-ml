# Computational Artifact Contract

## Same-phase reproducibility rule

Do not postpone important Python pipeline code to a later milestone.

If a phase computes something required for future regeneration, the persistent Python implementation must be created during that same phase.

## Temporary development work

Temporary Python is allowed for investigation/debugging.

However, a phase must not depend on:

- Claude chat history;
- `/tmp` files;
- manually copied column lists;
- one-off notebook cells;
- command history as the only record of logic.

## Completion evidence

Before closing a computational phase, confirm:

- required Python files exist;
- Python syntax/import checks pass;
- input/output paths are explicit;
- validation behavior exists;
- non-zero failure status is used for contract violations;
- Concourse CI could invoke the Python entry points later.
