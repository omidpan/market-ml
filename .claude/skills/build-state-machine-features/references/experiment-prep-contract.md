# Phase 3 Handoff Contract

Phase 2 should leave two buildable variants on the exact same timestamp universe:

## Control
Same:
- common start
- timestamps
- sessions
- target logic
- split logic
- sequence eligibility

Features:
- existing market-ml baseline features only

## ACD-enhanced
Same as control, plus the approved ACD/state-machine feature columns.

## First experiment
Use one representative ACD configuration only:

`nvda__or5__atr14__a010__c020__rules-v2`

Use one explicit regime policy per run.

Do not compare all six OR/ATR configurations until the single-config pipeline is validated end-to-end.

## Phase 3 should not change
- model architecture
- label policy
- split methodology
- purge/embargo behavior
- sequence length
- stride
- training hyperparameters

The first experiment isolates only the incremental effect of ACD/state-machine features.
