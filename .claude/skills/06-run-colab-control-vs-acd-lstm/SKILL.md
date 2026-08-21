---
name: 06-run-colab-control-vs-acd-lstm
descri~ption: Build the reproducible Google Colab GPU handoff and run contract for the paired NVDA CONTROL-vs-ACD LSTM experiment while keeping TEST sealed.
---

# 06 — Run Colab CONTROL vs ACD LSTM

## Purpose

Phase 06 turns the completed Phase-5 integrity checkpoint into one controlled,
reproducible GPU experiment:

- CONTROL = existing baseline feature set only;
- ACD = the same experiment plus the approved state-machine feature set;
- all labels, sequence keys, splits, model/training settings, and evaluation
  rules remain frozen and identical;
- the only intended experimental difference is the model feature set;
- full LSTM training runs on **Google Colab GPU only**;
- TEST remains sealed throughout Phase 06.

This skill must create a persistent, reusable Colab training entry point and a
thin runbook/handoff. It must not rely on pasted one-off notebook Python as the
primary implementation.

---

## Operating model

- User = manager, approval authority, and human Colab executor.
- ChatGPT = SME/reviewer.
- Claude Code = implementation developer only.
- Google Colab GPU = only full LSTM training environment.
- Concourse = static/data-preparation validation only; no training.

Claude/local/Concourse must **never** run full CONTROL or ACD LSTM training.

Do not create `.agent` directories.
Do not use or inspect retired agent directories.

---

## Current mode

PLAN MODE FIRST.

Do not implement until the user says exactly:

`Approved—implement it`

Before that approval Claude may inspect only what is necessary and return a
concise plan.

Even after implementation approval:

- do not run model training locally;
- do not run model training from Claude;
- do not run model training in Concourse;
- do not evaluate TEST;
- do not expose TEST class distribution, predictions, confusion matrices, or
  performance metrics.

The user will invoke training manually in Google Colab.

---

# Frozen experiment contract

Phase 06 is a paired ablation experiment, not a new model-design phase.

Current approved identities:

- symbol: `nvda`
- shared sequence universe total: `1,259,396`
- TRAIN: `878,828`
- VALIDATION: `189,428`
- TEST: `191,140` — sealed
- CONTROL feature set: `core_v1`
- CONTROL feature count: `23`
- ACD feature set: `core_v1_acd_v1`
- ACD feature count: `150`
- shared label policy: `atr_relative_3class_v1`
- ATR period: `14`
- horizon: `15m`
- multiplier: `0.75`
- current label-policy signature begins `cf167954f3b2`

Do not duplicate or rebuild the paired universe merely for Colab.
Consume the already approved Phase-3/Phase-4 artifacts.

Do not hardcode training hyperparameters into a new runner if the committed
CONTROL/ACD generated configs already carry them. Load them and verify equality.

The only approved CONTROL-vs-ACD semantic difference is the feature-set
identity/list already certified by Phase 05.

Follow `references/paired-experiment-contract.md`.

---

# Required persistent entry point

Preferred repository entry point:

`scripts/run_phase6_colab_control_vs_acd.py`

Claude may propose a different name only if it gives a concrete repository
reason in Step 1. The final mechanism must still be persistent Python.

Required explicit modes:

- `--mode preflight`
- `--mode train-control`
- `--mode train-acd`
- `--mode compare-validation`

Do **not** create a training `--mode all` that silently trains both models.
Training should always be an explicit human action per variant.

The runner should orchestrate the existing trusted training implementation
rather than reimplementing LSTM/model logic.

Follow `references/phase6-runner-contract.md`.

---

# Colab-only training guard

Training modes must fail closed outside Google Colab GPU.

At minimum they must verify at runtime:

1. Google Colab environment is present;
2. a TensorFlow GPU device is visible;
3. committed training dependencies match the approved stack;
4. Drive/repository/input/output roots resolve;
5. Phase-5 integrity evidence is PASS;
6. Phase-3/Phase-4 paired inputs match the frozen experiment identity;
7. TEST has not been requested for evaluation.

Current approved training package pins:

- `tensorflow==2.20.0`
- `keras==3.13.2`

Do not silently upgrade them.

A lightweight Colab bootstrap notebook may be created if useful, but it must
only mount Drive/install requirements/call the persistent Python runner. It
must not contain a second independent training implementation.

Follow `references/runtime-preflight-contract.md` and
`references/colab-execution-contract.md`.

---

# Training execution rules

CONTROL and ACD must be two separate, auditable runs.

For both runs, prove identical:

- ordered sequence universe;
- split assignment;
- label-policy identity/signature;
- label values for TRAIN/VALIDATION keys;
- sequence length and stride;
- target horizon;
- scaler policy and TRAIN-only fitting rule;
- output dtype;
- architecture;
- optimizer/loss;
- epochs/batch size;
- early stopping;
- learning rate / weight decay;
- gradient clipping;
- random seed;
- mixed precision setting;
- worker/input-pipeline settings that can change numerics;
- any class weighting or sampling policy.

Do not tune CONTROL and ACD separately.
Do not change hyperparameters after looking at one variant's validation result.
If a runtime failure requires a training-stack change, stop and treat it as a
new reviewed experiment revision rather than silently continuing.

Train one variant at a time. Either order is acceptable if the order is
recorded; preferred default is CONTROL first, then ACD.

Follow `references/paired-experiment-contract.md`.

---

# Output and artifact contract

Phase 06 must use a new Phase-6-owned output namespace and must not overwrite
prior baseline or Phase-4 artifacts.

Claude must inspect the existing trainer/output conventions and propose the
smallest compatible Phase-6 root in Step 1.

Each variant must persist enough evidence to reproduce and audit the run,
including where supported by the existing trainer:

- model/checkpoint artifact;
- training history;
- training report;
- exact generated config snapshot or identity;
- feature-set identity and feature count;
- label-policy identity/signature;
- sequence/split counts;
- seed and training settings;
- TensorFlow/Keras/Python runtime versions;
- GPU device identity;
- start/end timestamps;
- run status and failure reason if incomplete.

Do not overwrite an existing completed run by default. Fail clearly unless the
user explicitly requests an overwrite/revision strategy.

Follow `references/artifact-and-report-contract.md`.

---

# Validation-only comparison

After both runs complete, `--mode compare-validation` must produce one compact
paired comparison report using **VALIDATION only**.

It may include metrics already produced by the trusted trainer for validation,
such as:

- validation loss;
- validation accuracy;
- per-class precision/recall/F1 if already available;
- macro/weighted F1 if already available;
- best epoch;
- early-stopping outcome;
- training duration.

Do not invent new evaluation logic merely for Phase 06 unless separately
approved.

The comparison report must explicitly state:

- CONTROL and ACD used the same sequence/label/split contract;
- feature-set difference was the intended experimental variable;
- TEST was not evaluated;
- Phase 06 does not make a final TEST-based model-selection claim.

Follow `references/validation-test-sealing-contract.md`.

---

# Concourse boundary

Do **not** add a Phase-06 Concourse training job.
Do **not** create `ci/concourse/phase06.yml` for training.
Do **not** add `train-control` or `train-acd` to `ci/concourse/pipeline.yml`.

The existing Phase 01→02→03→04→05 static/data-preparation chain remains the
pre-Colab boundary.

If a tiny static CI check for a newly created Phase-6 Python file is genuinely
needed, Claude must propose it in Step 1 and justify why it does not introduce
training/runtime coupling. Default preference: leave Concourse unchanged.

---

# Step 1 — Inspect and plan only

Inspect only the minimum needed:

1. `scripts/run_phase4_control_vs_acd.py`;
2. `src/train_model.py`;
3. generated CONTROL/ACD configs;
4. Phase-4 label-policy manifest/report and paired-input identities;
5. Phase-5 integrity JSON report;
6. `requirements-train.txt`;
7. prior proven Colab baseline run artifacts/report only as needed to preserve
   the known-good invocation and runtime contract;
8. current model/report output conventions.

Do not scan unrelated repository history.
Do not inspect TEST values.
Do not inspect large Parquet content unless a lightweight metadata/count check
is actually necessary.

Return a concise implementation plan with:

- exact files to create/modify;
- whether to reuse Phase-4 training invocation or add a thin Phase-6 wrapper;
- exact CLI contract;
- Colab-only runtime guard;
- input identity/preflight checks;
- Phase-6 output/report roots;
- exact CONTROL and ACD run IDs;
- validation-comparison output;
- thin Colab runbook/notebook plan, if any;
- lightweight local tests/static checks;
- confirmation that Concourse remains non-training;
- anything blocking a reproducible Colab run.

Then stop and wait for:

`Approved—implement it`

---

# Step 2 — Implement after approval

After approval only:

- create the persistent Phase-6 Python runner;
- reuse the trusted trainer rather than duplicating model code;
- create deterministic preflight/runtime manifests;
- add the minimum targeted tests needed for fail-closed behavior;
- create a concise Colab runbook and optionally a thin bootstrap notebook;
- wire Phase-6-owned output/report locations;
- implement validation-only paired comparison;
- py-compile/static-test locally.

Local validation must not import/run TensorFlow unnecessarily if static tests
can prove the behavior without it.

Do not train.
Do not run Concourse.
Do not inspect TEST values.

---

# Step 3 — Colab preflight handoff

After implementation, give the user the exact minimal Colab sequence:

1. start a GPU runtime;
2. mount Google Drive;
3. enter the repository;
4. install `requirements-train.txt`;
5. run Phase-6 `--mode preflight`;
6. stop and inspect the preflight PASS report before any training.

Preflight must prove the runtime/input contract but must not train.

Do not hide failures behind warnings. Any mismatch in the frozen paired
experiment must be a hard failure.

---

# Step 4 — Human-invoked paired training

Only after a clean Colab preflight does the user manually invoke:

1. `--mode train-control`
2. verify the CONTROL run completed and artifacts were persisted;
3. `--mode train-acd`
4. verify the ACD run completed and artifacts were persisted;
5. `--mode compare-validation`

The user may run the two variants in separate Colab sessions if necessary, but
the runner must record enough runtime/config identity to prove comparability.

TEST remains sealed.

---

# Completion gate

Skill 06 is complete only when:

1. a persistent Phase-6 Colab runner exists;
2. training modes refuse to run outside Colab GPU;
3. committed `requirements-train.txt` is the runtime dependency source;
4. Phase-5 integrity PASS is required/verified;
5. CONTROL and ACD consume the same approved paired sequence/label universe;
6. CONTROL uses 23 approved baseline features;
7. ACD uses 150 approved baseline+ACD features;
8. all non-feature training settings are proven identical;
9. Phase-6 artifacts live in a new non-overwriting namespace;
10. CONTROL full training completes on Colab GPU;
11. ACD full training completes on Colab GPU;
12. validation-only paired comparison is produced;
13. no hyperparameter tuning between variants occurred;
14. no local/Claude/Concourse full training occurred;
15. no Concourse Phase-06 training job was added;
16. TEST was neve~r evaluated or opened for model-performance analysis;
17. final report clearly states Phase-6 conclusion is validation-only.

---

# Required completion report

A. Files created/modified  
B. Phase-6 runner CLI and execution guard  
C. Colab runtime/dependency proof  
D. Frozen paired-input identity proof  
E. CONTROL run ID, feature count, config identity, status  
F. ACD run ID, feature count, config identity, status  
G. Frozen common training settings proof  
H. CONTROL training artifact summary  
I. ACD training artifact summary  
J. Validation-only paired metric comparison  
K. Runtime/version/GPU manifest for both runs  
L. Confirmation no separate tuning occurred  
M. Confirmation no local/Claude/Concourse training occurred  
N. Confirmation TEST remained sealed  
O. Phase-6 PASS/FAIL and blockers before any later TEST-selection gate
