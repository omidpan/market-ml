# Configuration Integrity Contract

## Source of truth

`config/pipeline.yaml` remains canonical and hand-maintained.

Generated Phase-4 configs are derived artifacts:

- `config/generated/pipeline_phase4_control.yaml`
- `config/generated/pipeline_phase4_acd.yaml`

Concourse may persist copies in a Phase-4-owned subtree, but those copies do
not replace the canonical config.

## Comparison method

Parse YAML and compare semantic paths/values. Ignore comments, key ordering,
and formatting.

Produce three artifacts in the audit report:

1. canonical -> CONTROL diff
2. CONTROL -> ACD diff
3. whitelist decision for every changed path

Do not hardcode a conclusion that only feature fields differ. Detect the real
diff first.

## Mandatory invariant

CONTROL and ACD must use identical sequence, target, split, label, scaler, and
training settings. The experiment variable is the feature set.

Expected CONTROL feature identity: `core_v1`.
Expected ACD feature identity: `core_v1_acd_v1`.

Any unrelated difference is FAIL until explicitly reviewed.
