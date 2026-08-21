# Runtime and Preflight Contract

## Dependency source of truth

Use committed `requirements-train.txt`.

Current approved training pins:

- TensorFlow 2.20.0
- Keras 3.13.2

Do not use local Conda as the dependency source of truth.
Do not `pip freeze` Colab and replace the committed requirement files.

## Colab guard

Training must verify:

- Google Colab environment is present;
- TensorFlow reports at least one GPU device;
- TensorFlow version is exactly the approved pin;
- Keras version is exactly the approved pin.

If the runtime is CPU-only or a version differs, stop before loading/training
model tensors.

## Input integrity

Preflight should verify lightweight identities, manifests, counts, config
fields, and paths. Avoid expensive full scans when existing manifests/reports
already prove the identity.

Required evidence includes:

- Phase-5 overall PASS;
- canonical/CONTROL/ACD config integrity still valid;
- shared Phase-3 sequence universe identity;
- shared Phase-4 label-policy identity/signature;
- CONTROL feature_set/count = core_v1 / 23;
- ACD feature_set/count = core_v1_acd_v1 / 150;
- expected split counts;
- output root exists or can be created.

## Runtime manifest

Persist at least:

- UTC timestamp
- Python version
- TensorFlow version
- Keras version
- visible GPU device name(s)
- selected mixed-precision setting
- repository path and commit identifier if available
- CONTROL/ACD config paths
- requirements-train file identity/hash if practical

The runtime manifest is provenance, not a reason to mutate dependencies.
