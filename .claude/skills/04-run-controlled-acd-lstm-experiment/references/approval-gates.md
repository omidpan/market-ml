# Approval Gates

## Gate 1 — Implementation

Claude begins in plan mode.

No implementation until the user says:

`Approved—implement it`

## Gate 2 — Full training

After implementation and lightweight/static validation, Claude must report the exact frozen experiment configuration.

No full CONTROL or ACD training until the user explicitly approves training.

## Gate 3 — TEST

Phase 4 does not evaluate TEST.

A later milestone must explicitly authorize sealed TEST evaluation after experiment/configuration selection is frozen.
