# ACD / Opening Range State-Machine Feature Strategy for LSTM

## 1. Purpose

This document defines an ACD-style Opening Range state machine that converts the trading strategy into machine-learning features for a 1-minute OHLCV LSTM model.

The goal is **not** to give the LSTM a simple BUY/SELL signal. Instead, the deterministic ACD engine should describe:

- where price is relative to the Opening Range;
- where price is relative to A/C levels;
- whether a level was crossed, rejected, or broken;
- how long the current state has persisted;
- whether the Opening Range is narrow or wide;
- whether the market regime favors continuation or reversal;
- how many ACD signals have occurred during the day;
- whether previous ACD signals are still valid, expired, invalidated, or failed;
- how strong the current ACD state is.

The LSTM can then learn how these states interact with OHLCV, volatility, EMA, Bollinger Bands, time-of-day, and other features.

---

# 2. Opening Range Definition

Use the first 10 minutes of the regular session as the Opening Range (OR).

For each trading day:

```text
OR_High = highest High of first 10 one-minute bars
OR_Low  = lowest Low of first 10 one-minute bars

OR_Width = OR_High - OR_Low
```

The Opening Range is fixed after the first 10 minutes. It must not change during the remainder of the session.

## A and C levels

Using daily ATR:

```text
AUp   = OR_High + 0.05 * Daily_ATR
ADown = OR_Low  - 0.05 * Daily_ATR

CUp   = OR_High + 0.10 * Daily_ATR
CDown = OR_Low  - 0.10 * Daily_ATR
```

Structure:

```text
                 CUp
                  |
                 AUp
                  |
              OR_High
                  |
           Opening Range
                  |
              OR_Low
                  |
                 ADown
                  |
                CDown
```

All levels should remain fixed for the trading day.

---

# 3. Important Data-Leakage Rule

The ACD state engine must only use information that was available at the current 1-minute bar.

The first 10 minutes are known only after the 10th bar closes.

Therefore:

- Do not calculate A/C levels using future bars.
- Do not use the day's final High/Low.
- Do not use future signal outcomes when creating the current feature.
- Do not calculate a signal's strength from future price movement.
- Any future outcome is used only to create the **training label** or retrospective performance statistics.

The ACD feature at time `t` must be reproducible in live trading at time `t`.

---

# 4. ACD Market-State Machine

The state machine should distinguish between **position**, **event**, **persistence**, and **outcome**.

## Core position states

Suggested states:

```text
OR_INSIDE
BETWEEN_OR_HIGH_AND_AUP
ABOVE_AUP
ABOVE_CUP

BETWEEN_ADOWN_AND_OR_LOW
BELOW_ADOWN
BELOW_CDOWN
```

A simpler ML representation can encode the current location as:

```text
+2 = above CUp
+1 = between AUp and CUp
 0 = inside Opening Range
-1 = between CDown and ADown
-2 = below CDown
```

Do not rely on this categorical state alone. Preserve the continuous distances as additional features.

---

# 5. Distance Features

For every minute calculate normalized distances:

```text
dist_AUp   = (Close - AUp)   / Daily_ATR
dist_ADown = (Close - ADown) / Daily_ATR

dist_CUp   = (Close - CUp)   / Daily_ATR
dist_CDown = (Close - CDown) / Daily_ATR
```

This allows the model to understand whether price is:

- far from a level;
- approaching a level;
- touching a level;
- just beyond a level;
- moving deeply beyond a level.

Normalization by ATR makes the feature more comparable across symbols and volatility regimes.

---

# 6. Cross Events

A position above AUp is not the same as **just crossing AUp**.

Therefore create event features.

Example:

```text
AUp_cross_up
AUp_cross_down

ADown_cross_up
ADown_cross_down

CUp_cross_up
CUp_cross_down

CDown_cross_up
CDown_cross_down
```

A compact representation can be:

```text
+1 = bullish/upward crossing
-1 = bearish/downward crossing
 0 = no crossing
```

Crossing should be detected using the previous bar and current bar, without looking ahead.

---

# 7. ACD Reversal / Rubber-Band State

A major part of the strategy is the idea that price can touch an A/C line and reject it.

Example:

```text
Price approaches AUp
        |
        v
     touches
        |
        v
     reverses
        |
        v
remains below AUp
```

This should become an explicit state.

Suggested event:

```text
AUp_rejection_event = 1
```

and equivalent events:

```text
ADown_rejection_event
CUp_rejection_event
CDown_rejection_event
```

The event should only become a confirmed rejection after a predefined confirmation rule.

For example, a rejection can require:

1. price reaches/touches the level;
2. price moves back through the level;
3. price remains on the rejection side for at least N bars or one defined Opening-Range unit.

The exact confirmation rule should be tested rather than assumed.

---

# 8. Rejection Persistence

The LSTM should know not only that a rejection happened, but how recently it happened and whether it remains active.

Create:

```text
AUp_rejection_active
AUp_rejection_age
```

and equivalent features for the other levels.

Example:

```text
Bar 1: rejection event      = 1
Bar 2: rejection active     = 1, age = 1
Bar 3: rejection active     = 1, age = 2
Bar 4: rejection active     = 1, age = 3
```

A useful additional feature is a decaying strength:

```text
rejection_strength = exp(-age / tau)
```

This prevents an old rejection from having the same influence as a fresh rejection.

---

# 9. Signal Definition

Every ACD event should create a signal record.

Example:

```text
Signal #1
Time: 10:17
Direction: UP
Level: AUp
Event: AUp breakout
Initial state: ABOVE_A
```

Another:

```text
Signal #2
Time: 10:43
Direction: DOWN
Level: AUp
Event: AUp rejection
Initial state: REVERSAL_DOWN
```

The system should maintain a daily signal counter:

```text
AUp_up_count
AUp_down_count

ADown_up_count
ADown_down_count

CUp_up_count
CUp_down_count

CDown_up_count
CDown_down_count
```

Also create aggregate counts:

```text
acd_total_signals_today
acd_bullish_signals_today
acd_bearish_signals_today
acd_reversal_signals_today
acd_breakout_signals_today
```

These are useful features because the model can learn that the market has already produced several signals and may be in a different state.

---

# 10. Signal Count Example

Suppose during one trading day:

```text
10:15  AUp breakout
10:38  AUp rejection
11:10  ADown rejection
13:25  AUp breakout
14:40  AUp rejection
```

The daily state could contain:

```text
acd_total_signals_today       = 5
acd_bullish_signals_today     = 2
acd_bearish_signals_today     = 3
acd_AUp_signal_count          = 4
acd_ADown_signal_count        = 1
acd_breakout_count            = 2
acd_reversal_count            = 3
```

The LSTM can therefore see the **history of the ACD system within the current session**.

---

# 11. Signal Lifecycle

A signal should not remain valid forever.

Every signal should have a lifecycle:

```text
CREATED
   |
   v
ACTIVE
   |
   +------> CONFIRMED
   |
   +------> EXPIRED
   |
   +------> INVALIDATED
   |
   +------> FAILED
   |
   +------> REGIME_REVERSED
```

This is important because a signal that was correct for 5 minutes but later completely reversed should not be treated as permanently valid.

---

# 12. Signal Validity Window

Define a maximum validity period.

For example:

```text
signal_validity_bars = N
```

N should be determined experimentally.

If an AUp breakout occurs at 10:15 and the signal is only valid for 15 minutes:

```text
10:15 -> signal created
10:16 -> active
10:17 -> active
...
10:30 -> expires
```

If price never confirms the expected behavior, the signal becomes:

```text
EXPIRED
```

This prevents old signals from polluting the current market state.

---

# 13. "Stay for at Least One Opening-Range Unit"

Your strategy can define persistence relative to the Opening Range.

For example, after a line break:

```text
price crosses AUp
        |
        v
must remain above AUp
for at least N bars
```

or:

```text
N = Opening Range duration
N = 10 minutes
```

This creates a stronger confirmation than simply checking whether one candle closed above the level.

A possible feature:

```text
AUp_breakout_persistence
```

Example:

```text
1
2
3
4
5
...
10
```

If price returns below AUp before confirmation, the breakout can be marked:

```text
FAILED_BREAKOUT
```

---

# 14. Failed Signals Must Be Explicit

This is one of the most important additions.

The model should know when the ACD strategy made a wrong prediction.

Examples:

```text
AUp breakout
    |
    v
price fails to continue
    |
    v
returns below AUp
    |
    v
FAILED_BULLISH_BREAKOUT
```

Or:

```text
AUp rejection
    |
    v
price initially reverses
    |
    v
later breaks strongly above AUp
    |
    v
FAILED_BEARISH_REJECTION
```

Create explicit features:

```text
acd_failed_signal
acd_failed_breakout
acd_failed_reversal
acd_invalidated_signal
acd_regime_reversal
```

---

# 15. Returning to the Opening Range

A particularly important invalidation is when price completely returns to the Opening Range.

Example:

```text
AUp breakout
     |
     v
price stays above AUp
     |
     v
price falls
     |
     v
returns inside OR
```

This should invalidate the original bullish ACD state.

Possible state:

```text
RETURNED_TO_OR
```

Feature:

```text
acd_returned_to_or = 1
```

This is different from a normal pullback.

The model should be able to distinguish:

```text
healthy pullback
```

from:

```text
breakout invalidation
```

---

# 16. Complete Regime Reversal

An even stronger invalidation occurs when the market changes direction completely.

Example:

```text
AUp breakout
      |
      v
bullish state
      |
      v
price returns to OR
      |
      v
breaks ADown
      |
      v
bearish regime
```

This should produce:

```text
acd_regime_reversal = 1
```

and the previous bullish signal becomes:

```text
REGIME_REVERSED
```

This is extremely valuable information for the LSTM.

It tells the model:

> The previous ACD thesis has failed and the market has now entered the opposite regime.

---

# 17. Penalizing Wrong ACD Signals

There are two separate concepts:

## A. Feature penalty

For the LSTM input, create a running ACD quality/state score.

For example:

```text
acd_score
```

A successful confirmed signal can increase the score.

A failed signal decreases it.

Example concept:

```text
successful breakout     +1.0
successful reversal     +1.0
weak/expired signal      0
failed signal           -1.0
strong regime reversal  -2.0
```

This is only an initial framework. The actual weights should be learned or optimized using training data.

Do not assume that `-2` is objectively correct.

## B. Training-label penalty

Do not directly modify the future price target because ACD was wrong.

The target should remain objective, for example:

```text
future_return_15m
```

or:

```text
direction_15m
```

Instead, use ACD failure as an input feature.

This keeps the target independent from the strategy and reduces the risk of creating circular labels.

---

# 18. ACD Quality Score

A more sophisticated version can be continuous:

```text
acd_quality_score
```

Example conceptual calculation:

```text
base signal strength
+ persistence
+ confirmation
+ favorable OR regime
- signal age
- failed signals
- invalidation
- regime reversal
```

The score should be normalized, for example:

```text
-1.0 to +1.0
```

or:

```text
-100 to +100
```

The exact formula should be tested.

The important idea is that the LSTM receives a **continuous representation of the current ACD conviction/state** rather than only BUY/SELL.

---

# 19. Wide vs Narrow Opening Range

Calculate:

```text
OR_width_ATR = OR_Width / Daily_ATR
```

This should be a continuous feature.

Then optionally create a regime:

```text
OR_WIDTH_NARROW
OR_WIDTH_NORMAL
OR_WIDTH_WIDE
```

The strategy hypothesis is:

```text
WIDE OR
    -> rejection/reversal may be more likely

NARROW OR
    -> breakout/continuation may be more likely
```

However, the model should be allowed to discover the relationship.

Do not hard-code:

```text
wide = 80% reversal
narrow = 80% breakout
```

unless the historical data actually supports those probabilities.

---

# 20. Empirical ACD Probabilities

A very strong future enhancement is to calculate probabilities from historical data.

For example:

```text
Condition:
OR_width_ATR = 0.35
AUp touched
volatility = high
time = 10:30

Next 15 minutes:

reversal     = 63%
continuation = 37%
```

This can become:

```text
AUp_reversal_probability
AUp_breakout_probability
```

These should be calculated only from training data when used as model features.

Never calculate them using the validation/test period.

Otherwise they can leak future information.

---

# 21. Bollinger Band State

Bollinger Bands should also be represented as states rather than simply three numerical lines.

Suggested features:

```text
BB_position
BB_middle_state
BB_middle_persistence
BB_width
BB_width_change
```

## BB position

Example:

```text
+2 = above upper band
+1 = upper half
 0 = near middle
-1 = lower half
-2 = below lower band
```

## Middle state

```text
+1 = price above middle
-1 = price below middle
```

## Middle persistence

```text
BB_middle_persistence
```

Example:

```text
above middle:
1
2
3
4
5
...
```

This tells the LSTM how long price has maintained the state.

## Bollinger width

```text
BB_width =
    (UpperBand - LowerBand) / MiddleBand
```

And:

```text
BB_width_change
```

indicates whether volatility is contracting or expanding.

---

# 22. EMA State

EMA20 should also be converted into a state representation.

Recommended features:

```text
ema20_distance_ATR
ema20_slope
ema20_position
ema20_persistence
```

For example:

```text
ema20_position:

+1 = price above EMA20
-1 = price below EMA20
```

and:

```text
ema20_persistence:

1, 2, 3, 4, 5...
```

This represents:

> Price has remained above/below EMA20 for N consecutive minutes.

---

# 23. Time Regime

Because regular-session behavior is different throughout the day, include:

```text
minutes_since_open
minutes_to_close
session_phase
```

Possible phases:

```text
OPENING
EARLY_SESSION
MID_SESSION
LATE_SESSION
CLOSE
```

A more precise representation is continuous:

```text
minutes_since_open
```

The model can learn the relationship itself.

---

# 24. Recommended Feature Groups

The final feature architecture can be organized as:

```text
RAW PRICE/OHLCV
----------------
returns
body %
upper wick
lower wick
volume change


VOLATILITY
----------------
True Range
ATR-normalized TR
rolling volatility
volatility regime


EMA STATE
----------------
EMA20 distance / ATR
EMA20 slope
EMA20 position
EMA20 persistence


OPENING RANGE / ACD
----------------
OR width / ATR
OR position
AUp distance
ADown distance
CUp distance
CDown distance

AUp cross
ADown cross
CUp cross
CDown cross

AUp rejection
ADown rejection
CUp rejection
CDown rejection

rejection age
breakout persistence
signal age

signals today
bullish signals today
bearish signals today
breakout count
reversal count

active signal
expired signal
invalidated signal
failed signal
returned-to-OR
regime reversal

ACD quality score


BOLLINGER STATE
----------------
BB position
BB middle state
BB middle persistence
BB width
BB width change


TIME REGIME
----------------
minutes since open
minutes to close
session phase
```

---

# 25. Do Not Give the LSTM Only a Strategy Signal

Avoid:

```text
ACD_SIGNAL = BUY / SELL
```

Instead give the LSTM the components:

```text
where is price?
how far from the level?
did it cross?
did it reject?
how long has the state existed?
how old is the signal?
how many signals already occurred?
did the previous signal fail?
did price return to OR?
did the market reverse regime?
is OR wide or narrow?
is volatility expanding?
what time of day is it?
```

This allows the LSTM to learn interactions such as:

> AUp breakout + narrow OR + expanding volatility + strong volume + EMA20 above + Bollinger upper-half persistence + early-session regime

versus:

> AUp breakout + wide OR + high volatility + repeated AUp rejection + previous failed breakout.

Those are very different situations even though both technically say "AUp breakout."

---

# 26. Signal Statistics Per Day

The state engine should maintain daily statistics such as:

```text
acd_total_signals_today
acd_bullish_signals_today
acd_bearish_signals_today

acd_breakouts_today
acd_reversals_today

acd_AUp_signals_today
acd_ADown_signals_today
acd_CUp_signals_today
acd_CDown_signals_today

acd_successful_signals_today
acd_failed_signals_today
acd_expired_signals_today
acd_invalidated_signals_today

acd_current_signal_age
acd_current_signal_direction
acd_current_signal_strength
```

These values reset at the beginning of each trading day.

Important:

The current-day counters are not leakage as long as they only include signals that have already occurred.

---

# 27. Example Daily ACD State

Imagine:

```text
09:30â€“09:39
Opening Range established

OR width = 0.42 ATR
Regime = relatively wide


10:05
Price crosses AUp
Signal #1 = bullish breakout

10:09
Price falls below AUp
Signal #1 = FAILED BREAKOUT

10:20
Price touches ADown
reverses upward
Signal #2 = bullish reversal

10:30
Price remains above ADown for 10 minutes
Signal #2 = CONFIRMED

11:15
Price crosses AUp
Signal #3 = bullish breakout

12:00
Price returns into Opening Range
Signal #3 = INVALIDATED

13:40
Price breaks ADown
Signal #4 = bearish breakout

14:00
Price continues below ADown
Signal #4 = CONFIRMED
```

At 14:00, the model could receive:

```text
acd_total_signals_today       = 4
acd_bullish_signals_today     = 3
acd_bearish_signals_today     = 1

acd_breakouts_today           = 3
acd_reversals_today           = 1

acd_failed_signals_today      = 1
acd_invalidated_signals_today = 1
acd_confirmed_signals_today   = 2

acd_current_direction         = -1
acd_current_state             = BELOW_ADOWN

OR_width_ATR                  = 0.42
OR_regime                     = WIDE
```

This gives the LSTM a much richer description of the trading day.

---

# 28. Important Distinction: Strategy vs Model

The ACD state machine should be **deterministic**.

Given the same OHLCV data:

```text
same input
    â†“
same ACD state
```

The LSTM should be probabilistic:

```text
ACD state
+ OHLCV
+ volatility
+ EMA
+ Bollinger
+ time regime
+ sequence
        â†“
     LSTM
        â†“
P(up 5m)
P(up 10m)
P(up 15m)
```

This separation makes the system much easier to debug.

---

# 29. Recommended LSTM Targets

Because the current 15-minute target is weak during regular session, test multiple horizons:

```text
5-minute direction
10-minute direction
15-minute direction
30-minute direction
```

Keep the same feature generation and walk-forward methodology.

Compare performance separately for:

```text
pre-market
regular session
after-hours
```

and, within regular session:

```text
09:30â€“10:00
10:00â€“11:30
11:30â€“14:00
14:00â€“15:30
15:30â€“16:00
```

This can reveal whether the problem is:

- target horizon;
- market regime;
- time of day;
- insufficient features;
- weak ACD relationship;
- or lack of predictive information in the underlying data.

---

# 30. Recommended Development Sequence

Do not add the complete system at once.

Run controlled experiments.

### Experiment 1

Current 21 features.

```text
Baseline
```

### Experiment 2

Add:

```text
EMA persistence
Time regime
```

### Experiment 3

Add:

```text
Opening Range
AUp
ADown
CUp
CDown
distance
position
cross events
```

### Experiment 4

Add:

```text
rejection state
breakout persistence
signal age
signal count
failed signals
invalidations
regime reversal
ACD quality score
```

### Experiment 5

Add:

```text
Bollinger state
```

Compare every experiment using exactly the same:

- train/validation/test periods;
- purge;
- embargo;
- walk-forward methodology;
- target definition;
- transaction assumptions.

This allows you to measure the incremental predictive value of the ACD state machine.

---

# 31. The Core Principle

The most important concept for this architecture is:

```text
INDICATOR
   â†“
STATE
   â†“
EVENT
   â†“
PERSISTENCE
   â†“
OUTCOME
   â†“
REGIME
```

For example:

```text
AUp
 â†“
price crosses AUp
 â†“
BREAKOUT_UP
 â†“
remains above AUp for 10 minutes
 â†“
CONFIRMED
 â†“
later returns to OR
 â†“
INVALIDATED
 â†“
eventually breaks ADown
 â†“
REGIME_REVERSED
```

This sequence contains far more information than:

```text
ACD = BUY
```

The LSTM should learn from the **sequence of states and transitions**.

---

# 32. Final Proposed Architecture

```text
                  1-MIN OHLCV
                       |
                       v
             â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
             â”‚ Feature Engine   â”‚
             â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                      |
       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
       |              |              |
       v              v              v
  Price/Volume    ACD State      BB/EMA State
       |          Machine             |
       |              |               |
       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                      |
                      v
               Time/Session State
                      |
                      v
              Feature Vector
                      |
                      v
              Sequence Builder
                      |
                      v
                 LSTM Model
                      |
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          v           v           v
       5-min       10-min       15-min
       P(up)       P(up)        P(up)
          |           |           |
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                      v
                 Decision Layer
```

The ACD state machine therefore becomes a **structured market-context feature generator**, not a replacement for the LSTM.

The most valuable next step is to implement the ACD state engine independently and generate a dataframe where **every 1-minute row contains the complete current ACD state, event, persistence, signal count, validity, failure, and regime information**. Then the LSTM can consume those columns exactly like the other 21 features.