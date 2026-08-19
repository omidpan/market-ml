---
name: 04-run-controlled-acd-lstm-experiment
description: Plan and implement the first controlled CONTROL-vs-ACD LSTM experiment with persistent Python pipeline entry points and Concourse CI orchestration.
---

# 04 — Controlled ACD LSTM Experiment

## Operating model

- User = manager: sets direction, priorities, and approval gates.
- ChatGPT = SME: architecture, experiment design, leakage review, reproducibility criteria, and result review.
- Claude Code = developer: implements approved Python code and Concourse CI YAML.
- Concourse CI = executor: later runs the pipeline independently of Claude.

Claude must not be treated as a runtime dependency.

## Current mode

PLAN MODE ONLY.

Do not train.
Do not run the full Concourse pipeline.
Do not perform expensive rebuilds.

Claude may inspect only what is necessary to produce the implementation plan.

Stop at the explicit approval gate:

`Approved—implement it`

---

# Phase-4 objective

Compare exactly two variants on the same sample universe:

- CONTROL: `core_v1` — 23 baseline features
- ACD: `core_v1_acd_v1` — 150 total features

The only intended experimental difference is the feature set.

Frozen Phase-3 sample counts:

- train = 878,828
- validation = 189,428
- test = 191,140
- total = 1,259,396

Frozen Phase-3 roots:

- `data/parquet/sequence_index_phase3_common_v1`
- `data/parquet/model_matrix_phase3_common_v1`
- `data/parquet/features_1m_acd_v1_encoded`

Do not redesign or rebuild Phase 3 unless a reproducibility defect is discovered.

---

# Mandatory Python-pipeline rule

If a computation is required to regenerate or validate data from Parquet, CSV, database tables, feature stores, sequence indexes, model matrices, model artifacts, or experiment outputs, that computation must exist in persistent repository Python code.

Preferred locations:

- `src/*.py`
- `scripts/*.py`
- `tests/*.py`

For Phase 4, create:

`scripts/run_phase4_control_vs_acd.py`

This Python entry point must be suitable for later invocation by Concourse CI.

Temporary Python may be used for investigation, but required pipeline logic must not exist only in Claude chat, `/tmp`, notebook cells, copied command history, or manual lists.

Do not make Bash the implementation layer.

If shell appears necessary, treat it only as thin CI task wiring around persistent Python entry points.

Follow:

- `references/python-pipeline-contract.md`
- `references/computational-artifact-contract.md`

---

# Mandatory Concourse CI rule

During Phase 4, create Concourse CI YAML alongside the Python pipeline code.

Required files:

- `ci/concourse/phase01.yml`
- `ci/concourse/phase02.yml`
- `ci/concourse/phase03.yml`
- `ci/concourse/phase04.yml`

The YAML files orchestrate persistent Python entry points.

They must not duplicate feature generation, sequence generation, label creation, training, or evaluation logic.

Claude's responsibility is limited to:

1. generating the YAML;
2. validating YAML syntax;
3. checking dependency order;
4. checking referenced Python entry points and paths;
5. running lightweight/static validation only.

Claude must NOT execute the full Concourse pipeline just to prove the YAML.

The user will later run Concourse independently. If failures occur, ChatGPT will review the logs and prepare targeted fixes for Claude.

Follow:

`references/concourse-ci-contract.md`

---

# Hard experiment controls

CONTROL and ACD must use identical:

- sequence rows and ordering;
- labels;
- label-policy parameters;
- train/validation/test assignments;
- purge/boundary behavior;
- sequence length;
- horizon;
- scope;
- stride;
- sessions;
- model architecture;
- optimizer;
- learning rate;
- weight decay;
- gradient clipping;
- dropout;
- batch size;
- epoch limit;
- early-stopping settings;
- random seed;
- class-weight methodology;
- runtime settings that materially affect optimization.

Do not tune CONTROL and ACD independently.

---

# TEST rule

The TEST split remains sealed.

During Phase 4:

- do not evaluate TEST model performance;
- do not inspect TEST predictions;
- do not use TEST to select ACD configuration;
- do not use TEST for hyperparameter decisions.

TRAIN + VALIDATION only.

---

# Label-policy rule

Resolve the shared label-policy handoff before training.

Use the existing approved policy:

- policy: `atr_relative_3class_v1`
- ATR period: 14
- multiplier: 0.75
- horizon: 15m

Attach/build it once against the Phase-3 common sample universe and use the exact same label artifact for CONTROL and ACD.

Do not alter label semantics.

Follow:

`references/label-policy-contract.md`

---

# Step 1 — Plan only

Inspect only what is necessary to determine:

1. the exact current `build_label_policy.py` interface;
2. how to attach the existing ATR14 × 0.75 policy to the Phase-3 common universe;
3. the exact current `train_model.py` interface;
4. the actual active baseline hyperparameters;
5. the persistent Python functions/files required for Phase 4;
6. the structure of `scripts/run_phase4_control_vs_acd.py`;
7. the Phase-01 to Phase-04 Concourse YAML dependency chain;
8. the validation-only comparison outputs;
9. output paths and run identities;
10. any missing persistent Python entry point required to make Phases 01–03 callable from Concourse.

Do not train.
Do not run Concourse.
Do not perform a full data rebuild.

The plan must explicitly list:

- Python files to create/modify;
- YAML files to create/modify;
- inputs and outputs of each Python entry point;
- PASS/FAIL condition for each CI phase;
- dependency order Phase 01 → 02 → 03 → 04;
- exact places where explicit user approval is required.

Stop and wait for:

`Approved—implement it`

---

# Step 2 — Implementation after approval

Only after approval:

- implement persistent Phase-4 Python pipeline code;
- create `scripts/run_phase4_control_vs_acd.py`;
- add minimal helper functions/modules if needed;
- create/update `ci/concourse/phase01.yml` through `phase04.yml`;
- create shared Phase-4 label-policy artifact;
- add validation/comparison logic;
- run lightweight Python/YAML/static checks;
- run Phase-4 preflight through Python only if lightweight;
- do not run full Concourse;
- do not start full model training yet.

---

# Step 3 — Training approval gate

Before training, report:

- shared-label validation result;
- exact CONTROL and ACD Python entry points;
- exact frozen hyperparameter table;
- run IDs/output paths;
- proof that sample identity remains paired;
- proof TEST remains sealed;
- Concourse YAML syntax/dependency validation status.

Wait for explicit training approval.

---

# Step 4 — Controlled training

Only after explicit training approval:

1. train CONTROL;
2. train ACD;
3. keep all non-feature settings fixed;
4. evaluate VALIDATION only;
5. produce the comparison artifacts;
6. do not inspect TEST.

---

# Phase-4 completion gate

Phase 4 is complete only when:

1. required persistent Python files exist;
2. `scripts/run_phase4_control_vs_acd.py` exists;
3. `ci/concourse/phase01.yml` through `phase04.yml` exist;
4. Concourse YAML syntax and dependency order are validated;
5. essential computation lives in Python, not YAML/Bash;
6. the shared label policy is validated;
7. CONTROL and ACD remain paired;
8. validation comparison is reproducible;
9. no step requires Claude chat history;
10. TEST remains sealed.

The full Concourse pipeline does not need to be executed by Claude for Phase 4 to be implemented correctly.

The user will later run Concourse independently.

---

# Final Phase-4 report

Return:

A. Python files created/modified  
B. Concourse YAML files created/modified  
C. Phase 01–04 CI dependency map  
D. shared label-policy identity/path  
E. label counts/class balance  
F. frozen training configuration  
G. CONTROL run identity/path  
H. ACD run identity/path  
I. sample counts used by both runs  
J. best epoch for each  
K. validation loss/accuracy/balanced accuracy/macro-F1  
L. confusion matrices  
M. per-class precision/recall/F1  
N. ACD minus CONTROL deltas  
O. training-stability notes  
P. confirmation TEST was not evaluated  
Q. Python entry points suitable for Concourse execution  
R. next-step recommendation
