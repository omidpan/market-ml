# Phase 3 Reproducible Runner Contract

Phase 3 is not complete until the ad-hoc build commands used during development
are packaged into a reusable repository script.

## Required reusable entry point

Create one script under:

`scripts/run_phase3_control_vs_acd.py`

Preferred CLI:

- `--mode prepare`
- `--mode control`
- `--mode acd`
- `--mode validate`
- `--mode all`

The script must orchestrate existing approved modules rather than duplicate their
internal logic.

## Required behavior

The runner must:

1. Build the shared Phase-3 sequence index from `features_1m_acd_v1`.
2. Fit TRAIN-only categorical vocabularies from all feature rows consumed by TRAIN sequences.
3. Persist `categorical_encoding_manifest.json`.
4. Build/refresh `features_1m_acd_v1_encoded`.
5. Build the CONTROL model matrix with only `core_v1` features.
6. Build the ACD model matrix with `core_v1` + approved encoded ACD features.
7. Run the paired-universe validation.
8. Print final output paths and PASS/FAIL summary.

## Reproducibility requirements

- No dependency on `/tmp/acd_features_list.txt`.
- No manually copied feature list.
- Derive the ACD model feature list programmatically from:
  - the approved numeric/boolean ACD whitelist;
  - the persisted categorical encoding manifest.
- Do not hard-code generated one-hot column names.
- Read config/policy identities from `config/pipeline.yaml` where already defined.
- Preserve separate Phase-3 output roots.
- Do not overwrite production `sequence_index` or production `model_matrix`.
- Do not train.
- Do not touch `data/raw`.
- Re-running the script must be deterministic and safe after deleting only Phase-3 derived outputs.

## Completion requirement

The Phase-3 completion report must include the exact one-line command for a clean rebuild:

`python scripts/run_phase3_control_vs_acd.py --mode all`

and a validation-only command:

`python scripts/run_phase3_control_vs_acd.py --mode validate`
