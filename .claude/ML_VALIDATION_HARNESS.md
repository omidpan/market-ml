# Minimal ML Validation Harness

## Purpose

Use this harness to validate each major ML pipeline stage before the user approves the next Concourse job. Keep the first implementation limited to data, features, and model evaluation.

## Authority

- Concourse executes pipeline stages and deterministic checks.
- Claude reads reports, explains evidence, and recommends a status.
- Only the user may approve or trigger the next stage.
- Claude must not change source files, data, thresholds, jobs, models, or reports while reviewing results.

## Status Values

Every validation report must return exactly one status:

- `PASS`: All mandatory checks succeeded.
- `FAIL`: At least one mandatory check failed.
- `REQUIRES_REVIEW`: Evidence is incomplete, ambiguous, or contains an important warning.

Claude's status is advisory and never counts as approval.

## Harness 1: Data Validation

Run before feature engineering.

Check:

1. Required columns exist.
2. Timestamps are valid, sorted, and unique within each symbol.
3. OHLC values are logically consistent.
4. Volume is nonnegative.
5. Missing bars, gaps, NaN values, and infinite values are counted.
6. Raw source files remain unchanged.

Expected report:

`reports/data-validation.json`

Passing this harness makes feature engineering eligible for human approval.

## Harness 2: Feature Validation

Run after feature engineering and before training.

Check:

1. Selected model features contain no unexpected NaN or infinite values.
2. Feature timestamps and target timestamps are correctly aligned.
3. Features use only information available at or before prediction time.
4. Scalers, imputers, encoders, selectors, and similar preprocessing objects are fitted using training data only.
5. Feature names, feature parameters, dataset version, and output row counts are recorded.

Expected report:

`reports/feature-validation.json`

Passing this harness makes baseline training eligible for human approval.

## Harness 3: Model Evaluation

Run after baseline training.

Check:

1. Training, validation, and test periods are chronological.
2. The final test period was not used for tuning or model selection.
3. The candidate is compared with a simple baseline using the same evaluation period.
4. Training and validation metrics, parameters, dataset version, feature version, and run ID are recorded in MLflow.
5. Basic walk-forward results are recorded.
6. The expected model and fitted scaler artifacts exist.

Expected report:

`reports/model-evaluation.json`

Passing this harness makes the candidate eligible for final human review. It does not authorize registration or deployment.

## Required Report Fields

Each JSON report should contain:

- `stage`
- `status`
- `runId`
- `gitCommit`
- `datasetVersion`
- `startedAt`
- `completedAt`
- `checks`
- `warnings`
- `blockingIssues`
- `artifactPaths`

Every item in `checks` should include its name, result, observed value, expected condition, and concise evidence.

## Review Workflow

1. Run one Concourse stage.
2. Generate its JSON validation report.
3. Stop before the next stage.
4. Invoke the matching Claude review skill with the report and log paths.
5. Inspect Claude's evidence and recommendation.
6. Approve or reject the next stage manually.

## First-Draft Boundary

Do not add automated promotion, deployment, model registration, advanced drift monitoring, portfolio optimization, or automatic approval to this first version.
