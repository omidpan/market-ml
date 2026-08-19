# market-ml — Claude Code Project Instructions

## Project role

`market-ml` is the machine-learning research repository for causal prediction of U.S. equity market behavior from market data and derived quantitative features.

Current primary research symbol: `NVDA`.

Current controlled sequence-model baseline: `LSTM`.

The ACD state-machine/rule-engine is a separate project. `market-ml` may consume validated causal state-machine artifacts, but must not reimplement or modify the ACD engine.

## Active Claude skills

The project is restarting its Claude skill system milestone-by-milestone.

Only skills explicitly listed here are active.

Current active skill:

```text
.claude/skills/understand-market-ml/SKILL.md
```

All older/pre-existing skills are legacy/obsolete for this workflow.

Do not invoke, rely on, inspect for instructions, or extend legacy skills unless the user explicitly reactivates one.

Do not create or use `.agent`.
Do not inspect retired agent directories unless the user explicitly asks.

## Immutable ground truth

`data/raw/` is immutable source truth.

Never modify, delete, rename, move, overwrite, repair, reformat, regularize, or backfill files in `data/raw/`.

Never rewrite source timestamps, symbols, OHLCV, WAP, bar count, or source metadata.

Reading and validating `data/raw/` is allowed.

All cleaned, canonical, regularized, resampled, featured, labelled, sequenced, model-ready, or experiment-tracking artifacts must be written to approved derived locations outside `data/raw/`.

If a requested operation would alter `data/raw/`, stop and report the conflict.

## Existing derived-data architecture

Inspect current code/artifacts before assuming an active path.

Known derived namespaces include:

```text
data/parquet/ohlcv_1m_observed/
data/parquet/ohlcv_1m_regularized/
data/parquet/resampled/
data/parquet/features_1m/
data/parquet/targets_1m/
data/parquet/label_policy/
data/parquet/sequence_index/
data/parquet/model_matrix/
data/models/
```

Do not silently overwrite an existing dataset or model run.

Prefer additive, versioned artifacts.

## External state-machine data

Validated ACD/state-machine artifacts are external to this repository:

```text
$HOME/acd_experiments_local/<symbol>/
```

Treat that tree as read-only.

For current NVDA integration research, the common state-machine modeling boundary is:

```text
2020-01-03
```

Do not confuse the state-machine environment-threshold fit cutoff with an ML training cutoff.

Never use retrospective state-machine outcome/evaluation fields as model inputs.

## Point-in-time correctness

For model decision timestamp `t`:

```text
feature_available_at <= t
```

Never use:

- future bars;
- future aggregates;
- centered windows;
- future state-machine outcomes;
- future MFE/MAE;
- future direction correctness;
- targets/labels as input features;
- validation/test data to fit preprocessing;
- full-history statistics for train-time transforms.

Higher-timeframe features become usable only when the underlying higher-timeframe bar is complete.

## Session/time-series rules

Preserve chronological order.

Do not use random predictive train/test splitting.

Document and respect:

- market-session boundaries;
- timestamp convention;
- logical trading date;
- continuity rules;
- purge/embargo;
- final holdout;
- sequence endpoint semantics.

Fit scalers, encoders, imputers, clipping bounds, selectors, class weights, and similar train-time transforms using training data only.

Rebuild sequences after any approved filtering/join that changes the modeling universe.

## Model strategy

LSTM is the current controlled sequential baseline.

Do not switch to XGBoost, decision trees, CNNs, Transformers, or another model family during an LSTM milestone unless the user explicitly approves a model-comparison milestone.

Model comparisons must use the same approved:

- feature universe;
- target;
- timestamps;
- splits/folds;
- purge/embargo policy;
- final holdout.

## Colab execution

Google Colab is used for full training and GPU-intensive experiments.

Local execution is appropriate for lightweight:

- schema inspection;
- data validation;
- feature engineering;
- unit tests;
- smoke tests;
- small analytical checks.

Do not launch full training without explicit approval.

## Experiment tracking architecture

MLflow tracking is intentionally decoupled from the training environment.

The local machine runs the containerized MLflow tracking stack, including `mlflow-api-srv`.

Google Colab cannot directly reach the local `mlflow-api-srv`.

Therefore Colab training must not require direct network access to the tracker.

### Tracker payload contract

The existing tracker API contract uses a JSON payload shaped like:

```text
task
barSize
experimentName
horizon
metadata
parameters
metrics
```

The local tracker API endpoint is:

```text
POST /log-run
```

Do not couple model training to the MLflow Python client or the local tracker service.

The API boundary is intentional and helps isolate MLflow/tracking dependencies from model-training dependencies.

### Offline tracking from Colab

For Colab runs, training should write a tracker-ready JSON payload as a durable run artifact in Google Drive.

Preferred location:

```text
data/models/<experiment-identity>/run_id=<run_id>/mlflow_run_payload.json
```

If current repository code establishes a better approved run-artifact location, preserve it.

Do not write tracker payloads to `data/raw/`.

The payload should contain all data required by the local sync process to later call `/log-run`.

A later dedicated sync milestone may:

1. discover unsynchronized payloads;
2. POST them to local `mlflow-api-srv`;
3. record the returned MLflow experiment/run/artifact identity in a receipt sidecar.

Do not build that sync application during Milestone 0.

Do not require `requests`/MLflow networking for a successful Colab training run.

### Model artifacts

Inspect the actual current run directory and metadata contract before changing model artifact behavior.

Do not assume older `MODELS_DIRECTORY` behavior is still authoritative if the current training pipeline uses `data/models/...`.

## Feast

Feast is planned, not assumed implemented.

Do not create Feast infrastructure unless the user approves a dedicated milestone.

## Docker

Docker contains local infrastructure.

Inspect Docker configuration read-only when needed to understand the current tracker/model-volume architecture.

Do not start, stop, rebuild, or modify Docker services without explicit approval.

## Credentials

Never print, expose, upload, or commit:

- `.env` contents;
- tokens;
- passwords;
- private keys;
- service-account files;
- tracker credentials.

Environment-variable names/contracts may be inspected without exposing values.

## Planning and approval

Every architectural, data, feature, target, sequence, model, tracking, or infrastructure task starts with inspection and planning.

Before editing:

1. inspect relevant files read-only;
2. explain the current implementation;
3. identify authoritative inputs/outputs;
4. identify timestamp/session semantics;
5. identify leakage risks;
6. list exact files proposed to be created/modified;
7. explain the proposed change;
8. explain tests/validation;
9. report baseline lightweight test status when applicable;
10. stop and wait.

Implementation requires the exact explicit approval:

```text
Approved—implement it
```

Approval applies only to the listed scope.

If scope changes materially, stop and request new approval.

## Change management

Never without separate explicit approval:

- commit;
- merge;
- push;
- publish;
- deploy;
- delete historical artifacts;
- rewrite raw data;
- modify production/cloud resources;
- execute destructive commands.

Preserve unrelated user changes.

After approved implementation, report:

1. files created/modified;
2. tests/commands executed;
3. test results;
4. generated artifacts;
5. remaining risks;
6. deviations from the approved plan.

## Documentation authority

Some documentation may describe older stages of `market-ml`.

When docs, code, tests, and artifacts disagree:

1. identify the conflict;
2. do not silently choose;
3. determine the authority for the active task;
4. report before changing behavior.

## Milestone 0

Use:

```text
.claude/skills/understand-market-ml/SKILL.md
```

Milestone 0 is the first active Claude milestone.

It is read-only project discovery first.

Its purpose is to map the actual current ML journey before the state-machine feature milestone.

Milestone 0 must not change feature, target, sequence, model, or training behavior.

## Response style

Be concise and outcome-focused.

Clearly separate:

- confirmed facts;
- assumptions;
- recommendations;
- unresolved questions.

Do not dump generated code, terminal transcripts, or diffs unless requested.
