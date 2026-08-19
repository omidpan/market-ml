# Tracking Architecture Reference

## Intentional dependency boundary

Model training and experiment tracking are intentionally decoupled.

The training environment should not require the MLflow Python client or access to the local tracker network.

## Existing API payload

The tracker helper constructs:

```json
{
  "task": "...",
  "barSize": "...",
  "experimentName": "...",
  "horizon": 15,
  "metadata": {},
  "parameters": {},
  "metrics": {}
}
```

and historically POSTs it to:

```text
<TRACKER_API_BASE_URL>/log-run
```

The helper does not upload Keras/pickle/metadata files directly; the local tracker resolves artifacts through its shared model path/volume.

## Current Colab constraint

Colab cannot directly reach the local `mlflow-api-srv`.

Therefore the current research direction is:

```text
training run
 -> durable JSON payload
 -> Drive/local synchronization
 -> later local POST
```

The JSON should preserve the existing API payload contract so a future sync app can post it without reconstructing experiment semantics.

## Preferred payload placement

Prefer the tracker payload beside the run it describes:

```text
data/models/<identity>/run_id=<run_id>/mlflow_run_payload.json
```

Benefits:

- payload cannot lose run identity;
- versioned model/scaler/metadata stay together;
- Google Drive naturally carries the payload from Colab to local;
- future sync can be idempotent;
- no need to mix tracker networking into training.

A future sync milestone may also write:

```text
mlflow_sync_receipt.json
```

with the server response (`experimentName`, `runId`, `artifactUri`) after a successful POST.

Milestone 0 only documents this architecture.
