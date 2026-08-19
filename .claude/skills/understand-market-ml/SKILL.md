---
name: understand-market-ml
description: Milestone 0 read-only discovery of the current market-ml data, feature, target, sequence, LSTM training, experiment-artifact, and offline-MLflow-tracking architecture.
disable-model-invocation: true
---

# Milestone 0 — Understand market-ml

## Goal

Build an evidence-based map of what `market-ml` actually does today.

This is the first active Claude milestone.

Older/pre-existing Claude skills are obsolete for this workflow. Do not read them for instructions or invoke them unless the user explicitly reactivates one.

Do not create or use `.agent`.

## Read first

Read completely:

1. repository `CLAUDE.md`;
2. `.claude/rules/data-ground-truth.md`;
3. repository `README.md`;
4. current architecture/research docs;
5. current `config/`, `src/`, `scripts/`, `tests/`, `docker/`, `data/`, `reports/`, and model artifacts relevant to the pipeline;
6. all references bundled with this skill.

Do not treat old skills as authority.

## Phase A — repository and data map

Inspect read-only:

```text
config/
src/
scripts/
tests/
docs/
data/
data/parquet/
data/models/
reports/
experiments/
docker/
.claude/
```

Trace the actual current data path:

```text
data/raw
 -> observed/canonical
 -> regularized, if active
 -> resampled
 -> features_1m
 -> targets_1m / label_policy
 -> sequence_index
 -> model_matrix
 -> training
 -> run artifacts
```

For each layer report:

- physical path;
- grain;
- partitioning;
- primary key;
- timestamp convention;
- session scope;
- observed/imputed semantics;
- responsible code;
- responsible tests;
- NVDA first/last date when cheaply queryable.

Never modify `data/raw`.

Do not load all Parquets when metadata/partition/schema inspection is enough.

## Phase B — feature, target, and sequence audit

Identify the active authoritative builders.

Resolve exactly:

```text
feature definitions
feature-set identity
target definition
target horizon
label-policy logic
sequence length
stride
session/continuity scope
prediction timestamp
sequence endpoint
chronological split
purge/embargo
final holdout
```

Do not infer runtime behavior from folder names when code is available.

## Phase C — LSTM baseline audit

Find the active trainer(s) and controlled NVDA baseline.

Report:

- trainer entry point;
- TensorFlow/Keras version contract if available;
- model input shape;
- active feature set;
- target;
- architecture;
- optimizer/loss;
- class weighting;
- scaler/encoder behavior;
- split logic;
- seeds;
- early stopping/checkpoint behavior;
- Colab command;
- model output hierarchy;
- reports;
- representative baseline run ID.

Do not run full training.

## Phase D — Colab notebook history

If the current Colab notebook is available, inspect it as evidence of the experiment journey.

Separate:

```text
historical experiment commands
current active baseline
diagnostic-only runs
sealed-test statements
```

Do not assume every notebook cell is current production behavior.

## Phase E — MLflow/tracker audit

Inspect:

- current tracking helper(s);
- `docker-compose.yml`;
- relevant Dockerfiles;
- `mlflow-api-srv`;
- tracker-client code;
- shared model-volume/path contract;
- run metadata JSON;
- any existing sync notebook/script.

Confirm the current intended architecture:

```text
Colab training
   -> writes model/scaler/metadata
   -> writes tracker-ready JSON payload
   -> Google Drive syncs artifact locally
   -> later local sync process POSTs payload
   -> mlflow-api-srv /log-run
   -> MLflow tracker
```

Important:

- direct POST from Colab to the local service is currently disabled/not required;
- do not mix MLflow client dependencies into training;
- do not change tracker API semantics in Milestone 0;
- do not build the sync app in Milestone 0.

If the attached/current notebook does not actually contain sync logic, report that clearly rather than assuming it does.

## Phase F — model-run metadata inventory

Inspect representative `data/models/.../run_id=...` directories.

Identify:

```text
best/final model file
scaler
feature metadata
training report
config snapshot
metrics
parameters
run identity
existing MLflow-related JSON, if any
```

Determine whether a tracker-ready payload already exists.

If not, recommend its future location, but do not implement it in Milestone 0.

Preferred future sidecar:

```text
mlflow_run_payload.json
```

inside the versioned run directory.

## Phase G — Feast and external ACD boundary

Confirm whether Feast is implemented or only planned.

Document external read-only ACD root:

```text
$HOME/acd_experiments_local/nvda/
```

and common NVDA state-machine integration start:

```text
2020-01-03
```

Do not build state-machine features yet.

## Required completion report

Before any write, report:

1. current repository map;
2. actual data lineage;
3. NVDA data/Parquet coverage;
4. active feature builder;
5. active target builder;
6. active sequence/model-matrix builder;
7. active LSTM baseline contract;
8. chronological split/purge/embargo behavior;
9. model artifact hierarchy and baseline run;
10. Colab training workflow;
11. current MLflow API/tracker architecture;
12. exact current Colab-to-MLflow limitation;
13. current JSON/metadata artifacts and what is missing;
14. Docker/shared-volume architecture;
15. Feast status;
16. current tests;
17. docs-vs-code conflicts;
18. unresolved questions;
19. proposed Milestone-0 inventory artifacts only.

Mark key findings as:

```text
CONFIRMED_FROM_CODE
CONFIRMED_FROM_ARTIFACT
CONFIRMED_FROM_TEST
DOCUMENTED_ONLY
INFERRED
UNKNOWN
```

Stop and wait for explicit approval.

## Optional Milestone-0 write after approval

Only after:

```text
Approved—implement it
```

create documentation/inventory artifacts only.

Preferred:

```text
reports/project_inventory/market_ml_inventory.json
reports/project_inventory/market_ml_inventory.md
```

Do not change training/model/data behavior.

## Completion gate

Milestone 0 is complete when we can reproduce the current NVDA modeling contract and identify the exact insertion point for the next state-machine-feature milestone without guessing.
