# Validation Comparison and TEST-Sealing Contract

## Phase-6 evaluation boundary

TRAIN is used for optimization.
VALIDATION is used for the paired CONTROL-vs-ACD comparison.
TEST remains sealed.

## Forbidden in Phase 06

Do not:

- call a test evaluator;
- generate TEST predictions;
- calculate TEST loss/accuracy/F1;
- build TEST confusion matrices;
- inspect TEST per-class performance;
- expose TEST label distribution for model-selection purposes;
- use TEST to choose CONTROL vs ACD;
- tune hyperparameters based on TEST.

The presence/count of TEST rows may be checked only as a structural integrity
fact already established by prior phases.

## Validation comparison

Prefer metrics already emitted by the trusted trainer. Do not create a new
metric suite unless there is a separately approved reason.

The paired report should include:

- CONTROL validation metrics;
- ACD validation metrics;
- ACD minus CONTROL deltas for directly comparable metrics;
- best epoch/early-stopping status;
- feature counts;
- common config/label/sequence identity proof;
- `test_evaluated: false`.

## Interpretation

Phase 06 can conclude whether ACD improves or degrades VALIDATION performance
under the frozen experiment contract.

It cannot make a final out-of-sample TEST claim. Any later TEST opening must be
a separate explicit gate after the user decides the selection policy.
