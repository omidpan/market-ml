---
name: prepare-control-vs-acd-experiment
description: Prepare the first controlled baseline-vs-ACD experiment by rebuilding sequences and model matrices on an identical timestamp universe, without training.
---

# Milestone 1 — Phase 3: Prepare Control vs ACD Experiment

## Purpose

Prepare the first controlled experiment comparing:

- re-baselined control: existing market-ml features only
- ACD-enhanced: same rows/sequences + approved `core_v1_acd_v1` features

This phase stops before model training.

## Preconditions

Phase 2 is complete and validated:

- `state_machine_features_1m` rebuilt
- `features_1m_acd_v1` rebuilt
- common modeling start = `2020-01-03`
- control and ACD feature universes have identical timestamps
- approved ACD feature whitelist only
- no forbidden/leaking columns
- one explicit ACD config and one explicit regime policy
- Phase 2 targeted tests pass

## Fixed first experiment

Symbol:

`nvda`

ACD config:

`nvda__or5__atr14__a010__c020__rules-v2`

Regime policy:

`moderate_or_anchored-r1`

Do not compare other OR/ATR configs in this phase.

## Core experimental rule

The only experimental difference must be the added ACD/state-machine features.

Control and ACD variants must keep identical:

- source timestamps
- common start
- session universe
- label policy
- target horizon
- split boundaries
- purge/embargo behavior
- sequence length
- stride
- sequence eligibility
- train/validation/test rows
- random seed/config where relevant

## Required workflow

### Step 1 — Plan only

Inspect only the current sequence/model-matrix interfaces and the minimum configuration needed to rebuild both variants.

Do not rescan the repo.

Define exact commands and output identities for:

1. CONTROL
   - source features: existing `features_1m`
   - common start = `2020-01-03`
   - rebuilt sequence index
   - rebuilt model matrix
   - no ACD columns

2. ACD
   - source features: `features_1m_acd_v1`
   - same common start
   - rebuilt sequence index
   - rebuilt model matrix
   - approved ACD columns included

Stop and wait for:

`Approved—implement it`

### Step 2 — Build after approval

After approval:

- rebuild sequence outputs for both variants;
- rebuild model-matrix outputs for both variants;
- do not train;
- do not change target generation;
- do not change model architecture;
- do not change splits;
- do not alter Phase 2 feature tables.

### Step 3 — Validate paired-universe invariance

Run the checks in `references/paired-universe-test-contract.md`.

## Required completion report

Return only:

A. control dataset identity and paths  
B. ACD dataset identity and paths  
C. exact row/sequence counts by split for both  
D. exact proof that sequence timestamps/targets/splits are identical  
E. control feature count  
F. ACD feature count and incremental ACD feature count  
G. any categorical-encoding/materialization step performed  
H. any dropped rows and exact reason  
I. commands used  
J. whether Phase 4 training is unblocked

Do not train.
