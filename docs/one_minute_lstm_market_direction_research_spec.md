# One-Minute Market-Direction Research Pipeline: Technical Specification

> **Scope.** This document defines an offline, educational forecasting benchmark for one-minute OHLCV data. It is not investment advice, a trading signal, or a specification for live transaction execution.

**Data profile:** six years of consistently timestamped one-minute stock-candle data (OHLCV)  
**Candidate feature space:** 21 point-in-time, scale-stable indicators (normalized ratios and structural-candle features)  
**Prediction task:** classify the material direction of a future fixed-horizon move as **Up**, **Flat**, or **Down**.

---

## 0. Scope and point-in-time contract

This is a **one-minute intraday** modeling problem. It is high frequency relative to daily data, but it is not an HFT/market-microstructure system in the strict sense: it does not contain tick, quote, spread, or limit-order-book data. Evidence from those data sets is useful background, but it must not be treated as a direct result for this OHLCV-only pipeline.

For every decision timestamp **t**:

- Bar **t** is fully closed before features are calculated.
- Every input feature uses information available at or before the close of bar **t**.
- The target uses only prices observed after **t**.
- Each sequence contains exactly one symbol and one continuous market session; sequences never cross an overnight gap, trading halt, or market-closed period.

### Timestamp example

If the one-minute bar ending at **10:00:00** has just closed, the model may use its OHLCV values and a completed 30-minute bar covering **09:30–10:00**. It must not use a partially formed 30-minute bar covering **10:00–10:30**. The target begins only after the 10:00 signal time.

---

## 1. Architecture decision: LSTM is the primary baseline, not a presumed winner

An LSTM is a sensible first deep sequential model because it is compact, causal, and has an order-aware recurrent structure. It is **not** a proven universal winner over Transformers for one-minute stock OHLCV data. The chosen model must win a controlled, walk-forward comparison on this exact data and label definition.

| Model family | Role in the experiment | Required constraint |
| :--- | :--- | :--- |
| Majority-class and persistence baselines | Establish the minimum meaningful result. | Must be reported for every fold. |
| Logistic regression or tree-based tabular model | Tests whether sequence modeling adds value over lagged/aggregated features. | Same chronological folds and feature availability. |
| 1D-CNN | Tests local temporal-pattern extraction. | Causal input only. |
| Causal LSTM | Initial deep sequential baseline. | No future context; fixed point-in-time inputs. |
| Small causal Transformer | Direct challenger to the LSTM. | Causal mask plus positional or relative-position information. |
| CNN-LSTM hybrid | Optional later ablation. | Add only if simpler baselines are stable. |

### Correct interpretation of the LSTM-versus-Transformer question

- An LSTM encodes order through recurrence, but it does **not** guarantee that the last 1–3 minutes receive more importance than a point 60 minutes ago. Its gates learn what to retain.
- A causal Transformer does **not** assign equal weight to all time steps. It learns attention weights; positional/relative-position design and attention masking provide time-order information and can bias the model toward local context.
- Therefore, the correct design statement is: **start with an LSTM because it is a strong, efficient baseline; retain it only if it beats the alternatives consistently out of sample.**

Do not include raw absolute-price regression as the primary task. A persistence forecast can look numerically strong while having little directional value. The primary target is a normalized, fixed-horizon directional class.

---

## 2. Data and feature contract

### 2.1 Feature requirements

Each of the 21 features must be calculated per symbol and must be known at time **t**.

- Prefer returns, percentage distances, ATR-normalized ranges, relative candle-body/wick proportions, volume ratios, and bounded indicators such as RSI.
- Do not feed raw price levels or raw volume levels directly as ordinary numeric features.
- Treat “stationary” as a design goal, not an unproven guarantee. Verify stability with rolling distributions, feature-drift checks, and train-period-only scaling.
- Fit any scaler, imputation parameter, feature selector, or clipping bound on the training portion of the current fold only. Reuse it unchanged for that fold’s validation and test data.
- Use a causal rolling calculation only; never use centered windows or future global statistics.

If time-of-day, missing-bar flags, or symbol identifiers are added, update the feature count explicitly. The input is **(N, 120, 21)** only while the approved input count remains exactly 21.

### 2.2 Session, adjustment, and missing-data rules

- Define one market-session policy: regular hours only, or a separately labelled regular/extended-hours policy. Do not mix the two without an explicit feature/validation design.
- Use a consistent corporate-action adjustment policy across OHLC and volume.
- Never forward-fill across session boundaries, weekends, holidays, halts, or a material data outage.
- If an intraday bar is reconstructed for a data-quality study, retain a causal **is_imputed** indicator and report its frequency. Rebuild sequences after session segmentation.
- If a universal model pools symbols, maintain per-symbol ordering and use a common chronological cutoff across all symbols.

### 2.3 Higher-timeframe inputs

Higher-timeframe features may be used only after the underlying higher-timeframe candle is complete. The resampling implementation must align the feature to the first one-minute bar **after** the higher-timeframe close. Document the vendor timestamp convention and test this alignment with a timestamp example.

---

## 3. Sequence construction

Set the starting experiment configuration to:

| Parameter | Symbol | Initial value | Status |
| :--- | :--- | :--- | :--- |
| Lookback window | W | 120 completed one-minute bars | Starting hypothesis |
| Forecast horizon | H | 15 future one-minute bars | Starting hypothesis |
| Training anchor stride | S | 10 bars | Compute-saving choice |
| Feature count | F | 21 | Must match approved features |

For a sequence ending at time **t**:

$$
X_t = [z_{t-W+1}, z_{t-W+2}, ..., z_t]
$$

where **z_t** contains only features available after bar **t** closes. The batch tensor is:

$$
(samples, timesteps, features) = (N, 120, 21)
$$

For a continuous symbol-session segment with **T_j** valid rows, the number of usable anchors is:

$$
N_j = \max\left(0,\left\lfloor\frac{T_j-W-H}{S}\right\rfloor+1\right)
$$

and the full data-set count is:

$$
N = \sum_j N_j
$$

**Example:** a complete 390-minute regular session with **W=120**, **H=15**, and **S=10** produces:

$$
\left\lfloor\frac{390-120-15}{10}\right\rfloor+1=26
$$

usable anchors.

### Why these values are hypotheses, not fixed facts

- **W=120** provides two hours of context and is a reasonable compact starting point. It should be compared with shorter and longer causal windows, for example 60, 120, and 180 bars.
- **H=15** can reduce sensitivity to the next-bar noise problem, but the useful horizon is empirical and may vary by symbol, session, and volatility regime.
- **S=10** creates 90% fewer training anchors than **S=1**; however, adjacent retained windows still share **110 / 120 = 91.7%** of their input bars. It reduces compute and redundancy but does not make samples independent.
- Evaluation must match the intended prediction cadence. If results are reported every minute, use an every-minute evaluation data set; if results are reported every 10 minutes, evaluate at that cadence.

---

## 4. Volatility-adjusted direction labels

Use a scale-free future return and an ATR-relative threshold. All ATR values must be calculated using data available at or before **t**.

$$
r_{t,H} = \log\left(\frac{C_{t+H}}{C_t}\right)
$$

$$
a_t = \frac{\operatorname{ATR}_{14,t}}{C_t}
$$

$$
\tau_t = m \times a_t
$$

where **C_t** is the close of the fully observed signal bar, **C_{t+H}** is the close exactly `H` future one-minute bars later, and **m** is an ATR multiplier selected without using future validation or test outcomes.

| Condition | Class | Meaning |
| :--- | :--- | :--- |
| $r_{t,H} > \tau_t$ | $+1$ | Future movement exceeds the positive materiality threshold. |
| $|r_{t,H}| \leq \tau_t$ | $0$ | Future movement is within the volatility-scaled neutral band. |
| $r_{t,H} < -\tau_t$ | $-1$ | Future movement exceeds the negative materiality threshold. |

### Class-distribution rule

Do **not** force a universal 40/20/40 distribution. That ratio may be a useful diagnostic, but it is not a financial law and can turn the label threshold into a form of global data fitting.

Instead:

1. Select candidate multipliers from a predeclared grid using historical training data only.
2. Freeze the selected multiplier for the following validation period.
3. Report the natural class distribution for every fold and regime.
4. If imbalance is material, use training-set class weights or an appropriate loss function rather than redefining future outcomes to manufacture balance.

The neutral class should mean “not materially distinguishable under this research definition,” not “the fraction needed to improve an accuracy score.”

---

## 5. Causal LSTM baseline and permitted enhancements

### Initial causal baseline

Use a small baseline before adding complexity:

~~~text
(120, 21) input
    → feature normalization fitted on the training fold
    → causal LSTM
    → dropout / dense classification head
    → three logits: Down, Flat, Up
~~~

Tune hidden size, number of layers, regularization, learning rate, and early stopping only within the training/validation history available at each fold.

### Enhancement rules

- **Cell state and hidden state:** the network learns their roles. Do not assume that the cell state is inherently “multi-hour trend” memory while the hidden state is inherently “fast momentum” memory.
- **Forget-gate bias:** a small positive forget-gate initialization is a reasonable tested option. In Keras, **unit_forget_bias=True** already initializes the forget gate with **+1** by default.
- **Regularization:** compare ordinary input/output dropout, recurrent/variational dropout, and LayerNorm variants as ablations. Do not assume one is always superior; implementation choices can affect GPU speed and optimization behavior.
- **Stability controls:** use early stopping, gradient clipping, fixed seeds, and multiple-seed reporting.
- **Bidirectional LSTM:** do not use it for any predictive feature, model selection, scoring, or validation result in this pipeline. A reverse direction can use observations after **t**, invalidating a causal forecast. It is acceptable only for retrospective, non-predictive analysis whose outputs never enter this benchmark.

---

## 6. Leakage-resistant walk-forward validation

Random K-fold splitting is prohibited. The pipeline uses chronological folds and a final locked out-of-sample period.

### 6.1 Historical rolling folds

Use these initial durations:

| Segment | Initial duration | Purpose |
| :--- | :--- | :--- |
| Training window | 12 months | Fit transforms, labels, models, and class weights. |
| Purge/embargo gap | At least the maximum target horizon | Prevent targets from maturing across the boundary. |
| Validation window | 2 months | Compare configurations and tune using only prior history. |
| Forward step | 2 months | Advance the rolling window chronologically. |

For validation start **v_k** and embargo length **E**:

~~~text
[--------- 12-month training ---------][E gap][--- 2-month validation ---]
                                                  v_k
~~~

Advance **v_k** by two months for the next fold. If the rolling model uses a 15-minute horizon, **E** must be at least 15 minutes; increase it if any label, holding window, or cross-timeframe construction needs a longer separation.

### 6.2 Final test period

Reserve the newest meaningful block of history as a locked final out-of-sample test. Do not use it to choose the LSTM, the Transformer, the ATR multiplier, features, or thresholds. Report it once after the design is frozen.

### 6.3 Fold-isolation checklist

For every fold, fit or select the following using training data only:

- scalers, clipping limits, imputation rules, and feature selection;
- ATR-multiplier choice and class weights;
- model hyperparameters, checkpoint selection, and calibration;
- any symbol-level normalization parameters.

In addition, purge label overlap at fold boundaries and record the exact date/time boundaries, session policy, and sample counts.

---

## 7. Evaluation and model-selection rules

Report more than directional accuracy. A three-class model can obtain misleading accuracy through class imbalance.

| Category | Required reports |
| :--- | :--- |
| Discrimination | Balanced accuracy, macro-F1, Matthews correlation coefficient, per-class precision/recall, and a confusion matrix. |
| Probability quality | Cross-entropy/log loss, Brier score where applicable, and calibration plots. |
| Robustness | Mean and variation across chronological folds and random seeds; breakdowns by symbol, time of day, and volatility regime. |
| Baseline comparison | Improvement versus majority, persistence, and tabular baselines on the same held-out timestamps. |
| Reproducibility | Data-version identifier, feature list, session rule, fold boundaries, random seeds, and full configuration. |

Select a model only when it has a consistent advantage over simpler baselines across multiple forward folds **and** retains that advantage on the locked final test period. A single high-scoring fold is not sufficient evidence of a durable predictive relationship.

---

## 8. Design conclusions and experiment order

1. Build the point-in-time data contract and session-aware sequence builder.
2. Validate the ATR-normalized **Up / Flat / Down** labels and their natural class distribution.
3. Run the non-sequential and 1D-CNN baselines.
4. Run the causal LSTM baseline.
5. Compare it with a small causal Transformer under the same folds, labels, and feature set.
6. Add CNN-LSTM, LayerNorm, recurrent-dropout, or other hybrids only when a simpler model has demonstrated stable value.

This turns the original LSTM preference into a falsifiable research hypothesis: **the causal LSTM is retained because it wins on leakage-resistant, forward-looking evidence—not because the architecture is assumed to win in advance.**

---

## Evidence boundary

One electronic-trading comparison found LSTM-family models more robust for price-difference and movement tasks, but it used cryptocurrency limit-order-book data rather than one-minute stock OHLCV. A different high-frequency study reported stronger cumulative results for a Transformer hybrid. These are useful reasons to benchmark both families, not a basis for declaring an unconditional winner.

- [Transformers versus LSTMs for electronic trading](https://arxiv.org/abs/2309.11400)
- [Exploring the Advantages of Transformers for High-Frequency Trading](https://arxiv.org/abs/2302.13850)
- [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1808.03668)
- [Keras LSTM documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
