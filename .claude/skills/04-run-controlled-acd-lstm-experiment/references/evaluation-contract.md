# Phase 4 Validation-Only Evaluation Contract

Compare CONTROL and ACD on identical VALIDATION samples.

Required metrics:

- validation loss;
- accuracy;
- balanced accuracy;
- macro F1;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix;
- best epoch;
- epochs completed;
- train/validation loss trajectory;
- train/validation accuracy trajectory;
- early-stopping behavior.

Report absolute ACD minus CONTROL delta for main metrics and relative delta where meaningful.

All reproducible comparison calculations must exist in persistent Python code.

TEST remains sealed.
