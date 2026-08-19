# Concourse CI Contract — Phases 01–04

## Goal

Concourse CI should later be able to execute the numbered pipeline independently of Claude.

Required YAML:

- `ci/concourse/phase01.yml`
- `ci/concourse/phase02.yml`
- `ci/concourse/phase03.yml`
- `ci/concourse/phase04.yml`

## Core principle

Concourse YAML is orchestration only.

Persistent Python is the computational implementation.

Do not embed essential feature-generation, sequence-generation, labeling, training, or evaluation logic in YAML or Bash.

## Claude validation scope

Claude should:

- generate/update YAML;
- validate YAML syntax;
- verify referenced paths;
- verify Python entry points exist;
- verify dependency order;
- run lightweight/static checks.

Claude should NOT run the full Concourse pipeline.

Later, the user will run Concourse independently. Failures can be reviewed by ChatGPT and fixed through targeted Claude changes.

## Dependency order

`Phase 01 → Phase 02 → Phase 03 → Phase 04`

Later phases must not be treated as valid if required earlier gates fail.

## Phase 01

Phase 01 is primarily contract/design validation.

Do not invent heavy computation just to make it look like other phases.

PASS should verify the required integration/leakage contracts and any required static artifacts for Phase 02.

## Phase 02

CI should call the persistent Python responsible for state-machine feature construction/validation.

Expected important code includes:

- `src/state_machine_features.py`
- `tests/test_state_machine_features.py`
- any existing or newly needed persistent Phase-02 Python entry point

PASS should prove:

- approved causal whitelist;
- common modeling start;
- no forbidden outcome/evaluation columns;
- expected output schema;
- targeted tests pass;
- Phase-02 outputs can be generated or validated from Python.

## Phase 03

CI should call:

`scripts/run_phase3_control_vs_acd.py`

PASS should preserve the Phase-3 paired-universe contract.

Expected frozen values:

- CONTROL features = 23
- ACD features = 150
- incremental ACD = 127
- total samples = 1,259,396

The existing 26/26 paired-universe validation or maintained equivalent must remain the Phase-03 validation gate.

## Phase 04

CI should call:

`scripts/run_phase4_control_vs_acd.py`

Expected logical steps:

- labels;
- preflight;
- train-control;
- train-acd;
- compare;
- validate.

During development, training remains behind explicit user approval.

Before training, Phase-04 PASS/preflight requires:

- shared labels align;
- paired sample universe remains identical;
- frozen training configuration is confirmed;
- TEST remains sealed.

Final Phase-04 PASS requires successful controlled training and validation-only comparison after explicit approval.

## CI quality requirements

- explicit inputs/outputs;
- deterministic paths;
- non-zero exit on failure;
- no hidden state;
- raw data immutable;
- phase-owned outputs only;
- Python is the source of computational behavior;
- YAML only orchestrates.
