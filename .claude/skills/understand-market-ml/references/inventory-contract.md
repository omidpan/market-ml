# Milestone 0 Inventory Contract

If the user later approves inventory creation, write only project-discovery metadata.

Preferred files:

```text
reports/project_inventory/market_ml_inventory.json
reports/project_inventory/market_ml_inventory.md
```

The inventory must describe current state, not aspirations.

Required sections:

```text
repository
raw_data_contract
derived_data_layers
nvda_coverage
feature_pipeline
target_pipeline
sequence_pipeline
model_matrix_pipeline
lstm_training
colab_workflow
model_artifacts
tracking_api
offline_tracking_payload
docker
feast
tests
state_machine_boundary
documentation_conflicts
unresolved_questions
```

Every important claim should carry evidence status:

```text
CONFIRMED_FROM_CODE
CONFIRMED_FROM_ARTIFACT
CONFIRMED_FROM_TEST
DOCUMENTED_ONLY
INFERRED
UNKNOWN
```
