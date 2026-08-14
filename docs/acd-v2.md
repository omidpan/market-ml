ACD / Opening Range State-Machine Feature Specification

1. Purpose

This document defines how to convert the user’s ACD-style opening-range strategy into a structured state-machine feature layer for a 1-minute OHLCV LSTM model.

The goal is not to give the LSTM a simple BUY/SELL signal. Instead, the ACD layer should describe:

	●	the current opening-range regime;
	●	the relationship of price to A/CD levels;
	●	breakout and rejection events;
	●	persistence of each state;
	●	repeated signals during the day;
	●	invalidated or failed signals;
	●	regime changes;
	●	a confidence/strength score;
	●	penalties for signals that subsequently prove wrong.

The resulting state features can be combined with price action, volatility, EMA, Bollinger Band, volume, and time-regime features before being passed to the LSTM.



2. Opening Range Definition

The opening range is calculated from the first 10 regular-session 1-minute bars.

```text
OR_HIGH = highest high of first 10 minutes
OR_LOW  = lowest low of first 10 minutes
OR_WIDTH = OR_HIGH - OR_LOW
```

Daily ATR is used to normalize the A/C levels.

```text
A_UP   = OR_HIGH + 0.05 × Daily_ATR
A_DOWN = OR_LOW  - 0.05 × Daily_ATR
C_UP   = OR_HIGH + 0.10 × Daily_ATR
C_DOWN = OR_LOW  - 0.10 × Daily_ATR
```

Conceptually:

```text
                     C_UP
                       |
                     A_UP
                       |
                 OR_HIGH
                       |
               OPENING RANGE
                       |
                 OR_LOW
                       |
                    A_DOWN
                       |
                    C_DOWN
```

The exact percentages should remain configurable so they can be tested historically.



3. Why ACD Should Be a State Machine

A raw indicator tells the model where price is.

A state machine tells the model:

> What happened, what state are we currently in, how long has that state existed, and whether the previous signal is still valid.

For example, these two situations are not equivalent:

```text
Price is above A_UP
```

versus:

```text
Price crossed A_UP 2 minutes ago,
continued upward,
then rejected A_UP,
and has remained below it for 4 minutes.
```

The second contains substantially more information.

Therefore the ACD feature layer should represent state + event + persistence + outcome.



4. Core ACD States

A basic position state can be encoded as:

```text
+2 = above C_UP
+1 = between A_UP and C_UP
 0 = inside opening range
-1 = between A_DOWN and C_DOWN
-2 = below C_DOWN
```

This is a continuous market-location representation rather than a direct trading recommendation.

Suggested state names:

```text
INSIDE_OR
ABOVE_A
ABOVE_C
BELOW_A
BELOW_C
```



5. Distance Features

For every 1-minute bar, calculate normalized distance from the four A/C levels.

```text
dist_A_UP   = (Close - A_UP)   / Daily_ATR
dist_A_DOWN = (Close - A_DOWN) / Daily_ATR
dist_C_UP   = (Close - C_UP)   / Daily_ATR
dist_C_DOWN = (Close - C_DOWN) / Daily_ATR
```

These should generally be preferable to raw dollar distances because ATR normalizes different stocks and volatility regimes.



6. ACD Cross Events

Position and crossing are different features.

A crossing event occurs when price moves from one side of a level to the other.

For example:

```text
previous Close <= A_UP
current Close  >  A_UP
```

creates:

```text
A_UP_CROSS_UP = 1
```

A downward crossing creates:

```text
A_UP_CROSS_DOWN = -1
```

Equivalent events should be calculated for:

```text
A_UP
A_DOWN
C_UP
C_DOWN
```

A simple signed representation can be:

```text
+1 = bullish/upward cross
-1 = bearish/downward cross
 0 = no cross
```

Cross events should be event features, not permanent states.



7. Breakout State

A breakout is more than one bar crossing a line.

A useful breakout state should require confirmation, such as:

	1.	price crosses the level;
	2.	price remains beyond the level for N bars, or reaches a minimum normalized distance;
	3.	the move does not immediately return through the level.

The confirmation threshold should be configurable and tested.

Possible states:

```text
BREAKOUT_UP_PENDING
BREAKOUT_UP_CONFIRMED
BREAKOUT_DOWN_PENDING
BREAKOUT_DOWN_CONFIRMED
```

For example:

```text
A_UP crossed
      |
      v
PENDING
      |
      | stays above A_UP
      v
CONFIRMED BREAKOUT
```



8. Rubber-Band / Rejection State

A central part of the strategy is the idea that price can reach an A/C line, fail to continue, reverse, and remain away from the line.

This should be explicitly represented.

Example:

```text
Price approaches A_UP
        |
        v
Touches / crosses A_UP
        |
        v
Fails to continue
        |
        v
Moves back below A_UP
        |
        v
Remains below A_UP
        |
        v
A_UP REJECTION
```

Suggested event:

```text
A_UP_REJECTION_EVENT = 1
```

and an active state:

```text
A_UP_REJECTION_ACTIVE = 1
```

Equivalent states should exist for:

```text
A_DOWN
C_UP
C_DOWN
```



9. Rejection Persistence

A rejection should not remain active forever.

Create:

```text
rejection_age
```

Example:

```text
bar 1: rejection event
bar 2: age = 1
bar 3: age = 2
bar 4: age = 3
...
```

A rejection can become invalid if:

	●	price crosses back through the rejected level;
	●	price returns deeply into the opening range;
	●	an opposite breakout becomes confirmed;
	●	a configurable maximum age expires.

A decaying strength can also be calculated:

```text
rejection_strength = exp(-rejection_age / tau)
```

This gives the model a soft representation:

```text
fresh rejection     -> strong
older rejection     -> weaker
expired rejection   -> 0
```



10. Opening-Range Width Regime

Opening-range width is an important contextual feature.

Normalize it by daily ATR:

```text
OR_WIDTH_RATIO = OR_WIDTH / Daily_ATR
```

This allows comparison between different stocks and days.

Create both:

```text
OR_WIDTH_RATIO
```

and a categorical regime:

```text
NARROW
NORMAL
WIDE
```

The thresholds should be learned/calibrated from historical data rather than arbitrarily fixed.



11. Wide vs Narrow Opening Range Hypothesis

The strategy hypothesis is:

Wide Opening Range

A wide opening range may indicate that a large amount of early-session movement has already occurred.

Therefore:

```text
WIDE OR
   |
   +--> reversal/rejection may become more important
```

Narrow Opening Range

A narrow opening range may indicate compressed early-session movement.

Therefore:

```text
NARROW OR
   |
   +--> breakout/continuation may become more important
```

This should initially be treated as a hypothesis, not a hard-coded truth.

Do NOT initially assign:

```text
WIDE = 80% reversal probability
NARROW = 80% breakout probability
```

Instead provide:

```text
OR_WIDTH_RATIO
```

to the model and calculate empirical historical probabilities separately.



12. Empirical ACD Probability Features

A later version of the system can calculate conditional probabilities from historical training data.

Example:

```text
OR_WIDTH_RATIO = 0.35
A_UP touched
high volatility
10:15 AM
```

Historical outcome:

```text
reversal within next 15 minutes = 63%
continuation = 37%
```

This can become a feature such as:

```text
A_UP_REVERSAL_PROBABILITY
```

Likewise:

```text
A_UP_BREAKOUT_PROBABILITY
```

The probability must be generated using training data only.

It must not use future/test-period information.

This is particularly important to avoid leakage.



13. Signal Counting Per Day

The ACD engine should count signals, not only produce the current state.

Example:

```text
A_UP_SIGNAL_COUNT
A_DOWN_SIGNAL_COUNT
C_UP_SIGNAL_COUNT
C_DOWN_SIGNAL_COUNT
TOTAL_UP_SIGNALS
TOTAL_DOWN_SIGNALS
TOTAL_ACD_SIGNALS
```

Example trading day:

```text
A_UP signal #1
A_UP signal #2
A_DOWN signal #1
A_UP signal #3
```

The daily counters could therefore be:

```text
A_UP_COUNT   = 3
A_DOWN_COUNT = 1
```

This provides information about whether the market is repeatedly testing the same level.



14. Signal Identity

Every signal should receive an ID.

Example:

```text
Day: 2026-08-14
Signal #1:
A_UP breakout
Signal #2:
A_UP rejection
Signal #3:
A_DOWN breakout
```

Internally:

```text
signal_id
signal_type
signal_level
signal_time
signal_direction
signal_strength
signal_status
```

This makes it possible to evaluate each signal independently.



15. Signal Lifecycle

Every signal should have a lifecycle.

```text
CREATED
   |
   v
ACTIVE
   |
   +------------------+
   |                  |
   v                  v
CONFIRMED           INVALIDATED
   |
   +------------------+
   |                  |
   v                  v
SUCCESSFUL          FAILED
```

For example:

```text
A_UP_BREAKOUT
      |
      v
ACTIVE
      |
      | price remains above A_UP
      v
CONFIRMED
      |
      | target achieved
      v
SUCCESSFUL
```

But:

```text
A_UP_BREAKOUT
      |
      v
ACTIVE
      |
      | price immediately returns below A_UP
      v
INVALIDATED
```



16. Invalid Signal Definition

An invalid signal occurs when the market contradicts the original state before the signal has achieved its required objective.

Examples:

Failed bullish breakout

```text
Price crosses A_UP
       |
       v
Bullish breakout
       |
       v
Price falls back below A_UP
       |
       v
FAILED / INVALIDATED
```

Failed bearish breakout

```text
Price crosses A_DOWN
       |
       v
Bearish breakout
       |
       v
Price returns above A_DOWN
       |
       v
FAILED / INVALIDATED
```

Reversal failure

```text
A_UP rejection
      |
      v
Price begins reversing
      |
      v
Price crosses A_UP again
      |
      v
REJECTION INVALIDATED
```



17. Regime Change

The system must recognize that a valid bullish state can later become invalid.

Example:

```text
A_UP breakout
      |
      v
BULLISH REGIME
      |
      | price returns into OR
      v
NEUTRAL / RESET
      |
      | price breaks A_DOWN
      v
BEARISH REGIME
```

This is extremely important.

The system should never assume:

```text
BUY signal at 10:05
=
BUY state for the entire day
```

Instead, state must evolve.



18. ACD Regime State

A simple high-level regime can be:

```text
+2 = strong bullish
+1 = bullish
 0 = neutral / opening range
-1 = bearish
-2 = strong bearish
```

But the regime should be derived from actual state information.

For example:

```text
Strong bullish:
C_UP breakout + confirmation
Bullish:
A_UP breakout/hold
Neutral:
inside OR
Bearish:
A_DOWN breakdown/hold
Strong bearish:
C_DOWN breakdown + confirmation
```

The exact definitions should be validated experimentally.



19. Wrong-Signal Penalty

The ACD engine should explicitly measure when its state was wrong.

Create a signal score.

Example:

```text
+1.0  successful signal
+0.5  partially successful
 0.0  neutral/expired
-0.5  weak failure
-1.0  clear failure
```

A stronger version can use the actual future normalized move:

```text
signal_outcome =
    realized_move / Daily_ATR
```

However, this outcome is a training label/evaluation variable, not a live input feature.

It must not be fed into the model at prediction time.



20. Signal Strength vs Signal Outcome

Keep these separate.

Signal strength

Available at prediction time:

```text
OR_WIDTH_RATIO
distance_to_level
volume
volatility
persistence
cross_confirmation
time_regime
```

Signal outcome

Only available after the future has occurred:

```text
did_breakout_continue?
did_reversal_work?
maximum_favorable_excursion
maximum_adverse_excursion
future_5m_return
future_15m_return
```

This distinction prevents target leakage.



21. ACD Penalty Score

A live feature can represent the current historical/recent reliability of the ACD state, provided it is calculated only from information available up to the current bar.

Example:

```text
ACD_RELIABILITY_SCORE
```

It could be based on previous signals of the same type.

For example:

```text
Last 20 A_UP breakouts:
14 successful
6 failed
historical success rate = 70%
```

Then:

```text
A_UP_BREAKOUT_RELIABILITY = 0.70
```

This should be calculated with a rolling historical window and strictly causal data.



22. Recommended ACD Feature Set

A practical first version:

```text
1.  OR_WIDTH_RATIO
2.  OR_POSITION
3.  DIST_A_UP
4.  DIST_A_DOWN
5.  DIST_C_UP
6.  DIST_C_DOWN
7.  A_UP_CROSS
8.  A_DOWN_CROSS
9.  C_UP_CROSS
10. C_DOWN_CROSS
11. A_UP_REJECTION
12. A_DOWN_REJECTION
13. C_UP_REJECTION
14. C_DOWN_REJECTION
15. REJECTION_AGE
16. ACD_REGIME
17. ACD_REGIME_AGE
18. ACD_SIGNAL_COUNT_TODAY
19. A_UP_COUNT_TODAY
20. A_DOWN_COUNT_TODAY
21. PREVIOUS_SIGNAL_OUTCOME
22. ACD_RELIABILITY_SCORE
```

Additional features can be added later.



23. Important: Do Not Feed Future Outcome Into the LSTM

The following are useful for training/evaluation:

```text
signal_success
signal_failure
future_return
future_MFE
future_MAE
```

But they cannot be ordinary input features if they depend on future prices.

Instead:

```text
Historical ACD events
        |
        v
Outcome calculation
        |
        v
Training labels / statistics
```

while live model input is:

```text
Current ACD state
        |
        v
Current feature vector
        |
        v
LSTM
```



24. ACD + Bollinger State

The ACD state can be combined with Bollinger state.

Bollinger features:

```text
BB_POSITION
BB_MIDDLE_STATE
BB_MIDDLE_PERSISTENCE
BB_WIDTH
BB_WIDTH_CHANGE
```

For example:

```text
ACD = A_UP rejection
BB = below middle
EMA20 = falling
Volatility = high
```

This is much more informative than:

```text
ACD = SELL
```

The LSTM can learn the interaction between independent state systems.



25. EMA Persistence State

EMA20 should similarly become a state.

Suggested features:

```text
EMA20_POSITION
EMA20_DISTANCE_ATR
EMA20_SLOPE
EMA20_PERSISTENCE
```

Example:

```text
Price > EMA20
for 12 consecutive minutes
EMA20 slope positive
```

This tells the model that the market is in a persistent bullish state.



26. Time Regime

The ACD state should also be conditioned on time.

Useful features:

```text
MINUTES_SINCE_OPEN
MINUTES_TO_CLOSE
SESSION_PHASE
OPENING_PERIOD_FLAG
MIDDAY_FLAG
CLOSING_PERIOD_FLAG
```

The same ACD signal may behave differently at:

```text
09:35
11:30
14:00
15:50
```

Therefore time regime should be explicitly represented.



27. Final Feature Architecture

The overall system becomes:

```text
                     1-MIN OHLCV
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      Price/Volume     Volatility       Time Regime
          |                |                |
          +----------------+----------------+
                           |
                           v
                 +-------------------+
                 |   STATE ENGINE    |
                 +-------------------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
   ACD/Opening         Bollinger           EMA
      Range               State            State
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                  State Feature Vector
                           |
                           v
                  Sequence Construction
                           |
                           v
                         LSTM
                           |
              +------------+------------+
              |            |            |
              v            v            v
             5m           10m          15m
           target        target       target
```



28. Daily ACD Statistics

Every day, generate an ACD summary.

Example:

```text
Date: 2026-08-14
Opening Range:
High: ...
Low: ...
Width: ...
Width / ATR: ...
A_UP:
Signals: 3
Successful: 1
Failed: 2
Rejections: 2
Breakouts: 1
A_DOWN:
Signals: 2
Successful: 2
Failed: 0
Rejections: 1
Breakouts: 1
C_UP:
Signals: 1
Successful: 1
Failed: 0
C_DOWN:
Signals: 0
Final ACD Regime:
Bullish -> Neutral -> Bearish
Total signals:
8
Successful:
4
Failed:
3
Expired:
1
```

This gives you an interpretable daily diagnostic in addition to the ML features.



29. Why This Is Valuable for Your LSTM

The LSTM does not need to discover every trading concept from raw OHLCV.

Instead, it receives a structured representation:

```text
Raw market information
+
ACD state
+
ACD persistence
+
ACD signal history
+
ACD reliability
+
EMA state
+
Bollinger state
+
Time regime
```

Then the LSTM’s job is to learn:

> Given this sequence of market states, what is the probability of the next 5/10/15-minute direction?

This is a much cleaner ML problem.



30. Recommended Experiment Sequence

Do not add every feature at once.

Run controlled experiments:

Model A — Baseline

Current 21 features.

Model B — State enhancement

Add:

```text
EMA persistence
Time regime
```

Model C — ACD

Add:

```text
OR width
ACD position
ACD distances
cross events
rejection events
persistence
regime
signal count
```

Model D — ACD + reliability

Add:

```text
historical ACD success/reliability
```

Model E — Full state model

Add:

```text
Bollinger state
EMA state
ACD state
Time regime
```

Evaluate every model on exactly the same purged + embargoed walk-forward splits.



31. Metrics to Track

Do not evaluate only accuracy.

For each target horizon:

```text
5 minutes
10 minutes
15 minutes
```

track:

```text
AUC
Balanced Accuracy
Precision
Recall
F1
Calibration
Average future return
Profit factor
Maximum drawdown
```

And separately:

```text
Pre-market
Regular session
After-hours
```

For regular session, further break it down by time of day.



32. The Most Important Principle

The ACD engine should describe the market, not dictate the prediction.

Instead of:

```text
ACD says BUY
```

create:

```text
OR is narrow
Price crossed A_UP
Cross occurred 3 minutes ago
Price remains above A_UP
A_UP breakout persistence = 3
EMA20 slope = positive
Price above BB middle
BB bandwidth expanding
Volume increasing
Time since open = 25 minutes
```

The LSTM can then learn whether this combination historically predicts:

```text
5-minute direction
10-minute direction
15-minute direction
```

That preserves the information contained in your ACD strategy while allowing the ML model to discover when the strategy works and when it fails.



33. Key Design Rule

Do not hard-code the belief that ACD is always correct.

The model should be allowed to learn:

```text
ACD signal + market context
            |
            v
       sometimes works
       sometimes fails
```

Your proposed penalty / invalidation mechanism is therefore important.

A successful ACD state should strengthen confidence.

A failed ACD state should reduce confidence.

A regime change should reset or reverse the previous state.

That makes the ACD layer dynamic rather than a static indicator