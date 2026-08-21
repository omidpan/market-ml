# Artifact and Report Contract

## Namespace

Use a new Phase-6-owned output namespace. Do not overwrite:

- prior baseline model artifacts;
- Phase-3 matrix artifacts;
- Phase-4 label/config artifacts;
- Phase-5 integrity reports.

Claude must inspect current trainer conventions and choose the smallest
compatible Phase-6 model/report roots during planning.

## Variant isolation

CONTROL and ACD must have separate deterministic run directories/IDs.

A completed run should contain, directly or through the trusted trainer's
existing artifacts:

- run status
- model/checkpoint
- training history
- training report
- variant and feature-set identity
- feature count
- config snapshot/identity
- label-policy identity/signature
- sequence/split counts
- seed/training settings
- runtime manifest
- timestamps

## Overwrite policy

Default behavior: refuse to overwrite a completed run.

If the user intentionally reruns, use an explicit revision/run ID or explicit
overwrite flag approved by the user. Do not silently replace evidence from the
first run.

## Phase-6 comparison report

Persist a compact machine-readable JSON plus a human-readable summary containing
only validation metrics and provenance.

It must include an explicit field such as:

`test_evaluated: false`

and state that no final TEST-based selection has occurred.
