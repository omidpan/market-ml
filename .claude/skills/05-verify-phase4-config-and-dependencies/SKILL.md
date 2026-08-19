---
name: 05-verify-phase4-config-and-dependencies
description: Verify Phase-4 configuration integrity and audit the direct Python/Concourse/Colab dependencies before GPU training.
---

# 05 — Verify Phase-4 Configuration and Dependencies

## Purpose

This is the final pre-Colab integrity gate.

It must prove two things:

1. the generated Phase-4 CONTROL and ACD configs preserve the canonical
   `config/pipeline.yaml` experiment contract, with only approved feature-set
   differences between CONTROL and ACD;
2. the committed dependency files are sufficient for every persistent Python
   entry point invoked by Concourse Phases 01–05 and for later Colab GPU
   training, without copying the entire local Conda environment.

This phase does **not** train models.

---

## Operating model

- User = manager and approval authority.
- ChatGPT = SME/reviewer.
- Claude Code = implementation developer.
- Concourse CI = later independent executor.
- Google Colab GPU = only full LSTM training environment.

Claude/local/Concourse must not run full LSTM training.

---

## Current mode

PLAN MODE FIRST.

Do not implement until the user says exactly:

`Approved—implement it`

Do not run Concourse.
Do not run CONTROL/ACD training.
Do not inspect TEST model values/predictions.

---

# Gate A — Configuration-integrity proof

Compare semantically, not by text formatting/comments:

- `config/pipeline.yaml` — canonical source of truth
- `config/generated/pipeline_phase4_control.yaml`
- `config/generated/pipeline_phase4_acd.yaml`

The canonical config remains the hand-maintained source of truth.

Generated Phase-4 configs must remain reproducible derivatives. Never replace
the canonical config with an ACD-generated config.

## Required proof

Claude must identify the exact semantic diff among all three files.

CONTROL must preserve the canonical baseline experiment configuration unless
the Phase-4 generator has an explicitly approved mechanical reason to alter a
field.

ACD must differ from CONTROL only in the approved feature-set-related fields
required for `core_v1_acd_v1`.

At minimum prove identical CONTROL vs ACD values for:

- sequence length;
- target horizon;
- stride;
- scope;
- sessions;
- chronological split fractions/method;
- scaler policy;
- output dtype;
- label mode;
- label-policy name;
- ATR period;
- multiplier;
- neutral-threshold behavior;
- architecture;
- hidden size;
- layer count;
- dropout;
- batch size;
- epochs;
- learning rate;
- weight decay;
- gradient clipping;
- early-stopping monitor/patience/min_delta;
- seed;
- mixed-precision setting;
- worker setting;
- any class-weight/loss/optimizer fields present in the current config.

Expected intended feature difference:

- CONTROL: `feature_set=core_v1`, 23 model features
- ACD: `feature_set=core_v1_acd_v1`, 150 model features

Do not assume those are the only actual diffs. Compute and report the complete
semantic diff and fail if an unapproved difference exists.

Follow `references/config-integrity-contract.md`.

---

# Gate B — Dependency-integrity proof

Audit the actual direct external Python dependencies required by the persistent
Python entry points used by:

- Phase 01
- Phase 02
- Phase 03
- Phase 04
- Phase 05
- later Colab training

Do **not** use `pip freeze`.
Do **not** generate requirements from the entire `data-engineer` Conda environment.
Do **not** add unrelated packages merely because they are installed locally.

Inspect the final Python entry points and their relevant local-module import
closure.

Classify dependencies into the smallest useful committed sets.

Preferred structure if justified:

- `requirements.txt` — core/data pipeline runtime
- `requirements-ci.txt` — CI/test/static-validation additions
- `requirements-train.txt` — Colab/full-training additions

A clean alternative structure is acceptable if Claude proves it is simpler and
complete.

Concourse task files must install the correct committed dependency set for the
Python they invoke.

Colab must have a committed training dependency file; the local Conda
environment is not the source of truth.

Follow `references/dependency-audit-contract.md`.

---

# Persistent Python requirement

Because this phase performs repeatable computation over configs/imports, its
proof must live in repository Python, not only in Claude chat or pasted shell.

Preferred single entry point:

`scripts/run_phase5_integrity_audit.py`

Recommended modes:

- `--mode config`
- `--mode dependencies`
- `--mode concourse`
- `--mode all`

It should:

- fail non-zero on contract violations;
- produce deterministic PASS/FAIL output;
- write a compact machine-readable audit report under a Phase-5-owned report
  directory;
- avoid modifying Parquet/model artifacts;
- be callable locally and later by Concourse.

Follow `references/python-audit-runner-contract.md`.

---

# Concourse Phase 05

Create:

`ci/concourse/phase05.yml`

and extend:

`ci/concourse/pipeline.yml`

with:

`Phase 01 -> Phase 02 -> Phase 03 -> Phase 04 -> Phase 05`

Phase 05 is validation only.

It may invoke the Phase-5 Python audit and lightweight tests/import checks.

It must not train.
It must not invoke `train-control` or `train-acd`.
It must not import TensorFlow merely to prove CI wiring unless a lightweight
training-dependency import smoke test is explicitly designed and justified.

Do not run the real Concourse pipeline during implementation.

Only validate:

- YAML syntax;
- declared resources/jobs;
- `passed:` dependency order;
- task input/output mappings;
- Python entry-point wiring;
- dependency-file selection.

Follow `references/concourse-phase05-contract.md`.

---

# Colab handoff readiness

Skill 05 does not run Colab training.

It must leave enough evidence to start the next Colab skill safely:

- canonical/CONTROL/ACD config-integrity PASS;
- dependency audit PASS;
- committed training dependency file exists;
- Phase-4 shared labels exist and are unchanged;
- Phase-3 paired universe remains the approved source;
- TEST remains sealed;
- no local/Claude/Concourse training has occurred.

Follow `references/colab-readiness-contract.md`.

---

# Step 1 — Plan only

Inspect only what is necessary:

1. current `config/pipeline.yaml`;
2. generated CONTROL/ACD Phase-4 configs;
3. persistent Python entry points used by Phase 01–04;
4. relevant local imports reachable from those entry points;
5. current requirements files;
6. `ci/concourse/pipeline.yml` and `phase01.yml`–`phase04.yml`;
7. the Phase-4 clean-run/preflight evidence already present in the repo/logs
   if needed.

Return a concise implementation plan containing:

- exact files to create/modify;
- exact config-diff rules;
- exact dependency classification;
- proposed requirements file structure;
- proposed Phase-5 Python audit entry point;
- proposed `phase05.yml` and pipeline dependency extension;
- lightweight validation to run;
- anything that would block Colab readiness.

Then stop and wait for:

`Approved—implement it`

---

# Step 2 — Implement after approval

After approval only:

- create the persistent Phase-5 audit Python entry point;
- produce semantic config-diff validation;
- audit and update committed dependency files;
- update Concourse task dependency-file usage if required;
- create `ci/concourse/phase05.yml`;
- extend `ci/concourse/pipeline.yml` through Phase 05;
- add targeted tests where useful;
- run local lightweight validation only.

Do not run Concourse.
Do not train.

---

# Completion gate

Skill 05 is complete only when:

1. canonical vs CONTROL vs ACD semantic diff is reported;
2. every unapproved config difference is zero;
3. CONTROL and ACD training/sequence/label settings are proven identical;
4. CONTROL feature count is 23 and ACD feature count is 150, or a changed
   approved count is explicitly explained before proceeding;
5. dependency files cover the actual Phase 01–05 direct requirements;
6. no `pip freeze`/whole-Conda dump was used;
7. Concourse tasks point to the appropriate committed requirement files;
8. `phase05.yml` exists and is wired after Phase 04;
9. YAML/dependency wiring passes static validation;
10. no Concourse pipeline execution occurred;
11. no model training occurred;
12. TEST remains sealed;
13. the next Colab training phase is unblocked.

---

# Required completion report

A. Files created/modified  
B. Canonical → CONTROL semantic diff  
C. CONTROL → ACD semantic diff  
D. Approved vs unexpected differences  
E. Frozen common training/sequence/label settings  
F. CONTROL/ACD feature counts  
G. Direct external dependency inventory by phase  
H. Final requirements-file structure and contents/categories  
I. Concourse task → requirements-file mapping  
J. Phase 01→02→03→04→05 dependency proof  
K. Tests/static checks and results  
L. Confirmation no Concourse run occurred  
M. Confirmation no training occurred  
N. Confirmation TEST remains sealed  
O. Colab-readiness PASS/FAIL and blockers
