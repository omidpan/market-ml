
import numpy as np
import pandas as pd

def create_candle_test_scenarios() -> pd.DataFrame:
    scenarios = [
        # ---------------------------------------------------------
        # Normal candles
        # ---------------------------------------------------------
        {
            "Scenario": "Normal bullish",
            "open": 100.00,
            "high": 106.00,
            "low": 98.00,
            "close": 104.00,
            "volume": 250000,
        },
        {
            "Scenario": "Normal bearish",
            "open": 104.00,
            "high": 106.00,
            "low": 98.00,
            "close": 100.00,
            "volume": 220000,
        },

        # ---------------------------------------------------------
        # Open and close equal or extremely close
        # ---------------------------------------------------------
        {
            "Scenario": "Exact doji with positive volume",
            "open": 100.00,
            "high": 105.00,
            "low": 95.00,
            "close": 100.00,
            "volume": 150000,
        },
        {
            "Scenario": "Tiny bullish body",
            "open": 100.0000,
            "high": 100.0100,
            "low": 99.9900,
            "close": 100.0001,
            "volume": 12000,
        },
        {
            "Scenario": "Tiny bearish body",
            "open": 100.0001,
            "high": 100.0100,
            "low": 99.9900,
            "close": 100.0000,
            "volume": 11000,
        },

        # ---------------------------------------------------------
        # Open or close close to high
        # ---------------------------------------------------------
        {
            "Scenario": "Open almost equals high",
            "open": 100.0000,
            "high": 100.0001,
            "low": 95.0000,
            "close": 97.0000,
            "volume": 80000,
        },
        {
            "Scenario": "Close almost equals high",
            "open": 97.0000,
            "high": 100.0001,
            "low": 95.0000,
            "close": 100.0000,
            "volume": 85000,
        },
        {
            "Scenario": "Open exactly equals high",
            "open": 105.00,
            "high": 105.00,
            "low": 98.00,
            "close": 100.00,
            "volume": 90000,
        },
        {
            "Scenario": "Close exactly equals high",
            "open": 100.00,
            "high": 105.00,
            "low": 98.00,
            "close": 105.00,
            "volume": 95000,
        },

        # ---------------------------------------------------------
        # Open or close close to low
        # ---------------------------------------------------------
        {
            "Scenario": "Open almost equals low",
            "open": 100.0001,
            "high": 105.0000,
            "low": 100.0000,
            "close": 103.0000,
            "volume": 75000,
        },
        {
            "Scenario": "Close almost equals low",
            "open": 103.0000,
            "high": 105.0000,
            "low": 100.0000,
            "close": 100.0001,
            "volume": 70000,
        },
        {
            "Scenario": "Open exactly equals low",
            "open": 100.00,
            "high": 107.00,
            "low": 100.00,
            "close": 105.00,
            "volume": 120000,
        },
        {
            "Scenario": "Close exactly equals low",
            "open": 105.00,
            "high": 107.00,
            "low": 100.00,
            "close": 100.00,
            "volume": 115000,
        },

        # ---------------------------------------------------------
        # Full-body candles
        # ---------------------------------------------------------
        {
            "Scenario": "Full-body bullish",
            "open": 100.00,
            "high": 105.00,
            "low": 100.00,
            "close": 105.00,
            "volume": 300000,
        },
        {
            "Scenario": "Full-body bearish",
            "open": 105.00,
            "high": 105.00,
            "low": 100.00,
            "close": 100.00,
            "volume": 280000,
        },

        # ---------------------------------------------------------
        # Large shadows
        # ---------------------------------------------------------
        {
            "Scenario": "Long upper shadow",
            "open": 100.00,
            "high": 120.00,
            "low": 99.00,
            "close": 102.00,
            "volume": 400000,
        },
        {
            "Scenario": "Long lower shadow",
            "open": 100.00,
            "high": 103.00,
            "low": 80.00,
            "close": 102.00,
            "volume": 420000,
        },
        {
            "Scenario": "Symmetrical shadows",
            "open": 99.00,
            "high": 105.00,
            "low": 95.00,
            "close": 101.00,
            "volume": 180000,
        },

        # ---------------------------------------------------------
        # Extremely small ranges
        # ---------------------------------------------------------
        {
            "Scenario": "Extremely small bullish range",
            "open": 100.00001,
            "high": 100.00004,
            "low": 100.00000,
            "close": 100.00003,
            "volume": 2_500,
        },
        {
            "Scenario": "Extremely small bearish range",
            "open": 100.00003,
            "high": 100.00004,
            "low": 100.00000,
            "close": 100.00001,
            "volume": 2400,
        },

        # ---------------------------------------------------------
        # Flat candles
        # ---------------------------------------------------------
        {
            "Scenario": "Flat candle with positive volume",
            "open": 100.00,
            "high": 100.00,
            "low": 100.00,
            "close": 100.00,
            "volume": 25000,
        },
        {
            "Scenario": "Flat candle with zero volume",
            "open": 100.00,
            "high": 100.00,
            "low": 100.00,
            "close": 100.00,
            "volume": 0,
        },

        # ---------------------------------------------------------
        # Volume cases
        # ---------------------------------------------------------
        {
            "Scenario": "Normal candle with zero volume",
            "open": 100.00,
            "high": 105.00,
            "low": 98.00,
            "close": 103.00,
            "volume": 0,
        },
        {
            "Scenario": "Normal candle with very small volume",
            "open": 100.00,
            "high": 105.00,
            "low": 98.00,
            "close": 103.00,
            "volume": 1,
        },

        # ---------------------------------------------------------
        # Zero-price denominator cases
        # ---------------------------------------------------------
        {
            "Scenario": "Zero open",
            "open": 0.00,
            "high": 0.02,
            "low": 0.00,
            "close": 0.01,
            "volume": 5000,
        },
        {
            "Scenario": "Zero close",
            "open": 0.01,
            "high": 0.02,
            "low": 0.00,
            "close": 0.00,
            "volume": 4000,
        },

        # ---------------------------------------------------------
        # Price-scale cases
        # ---------------------------------------------------------
        {
            "Scenario": "Very small price",
            "open": 0.00010,
            "high": 0.00012,
            "low": 0.00009,
            "close": 0.00011,
            "volume": 5000000,
        },
        {
            "Scenario": "Very large price",
            "open": 999_999.00,
            "high": 1000010.00,
            "low": 999_990.00,
            "close": 1000001.00,
            "volume": 50,
        },

        # ---------------------------------------------------------
        # Invalid OHLC cases
        # ---------------------------------------------------------
        {
            "Scenario": "Invalid: high below open",
            "open": 105.00,
            "high": 103.00,
            "low": 98.00,
            "close": 100.00,
            "volume": 20000,
        },
        {
            "Scenario": "Invalid: low above close",
            "open": 105.00,
            "high": 110.00,
            "low": 102.00,
            "close": 100.00,
            "volume": 20000,
        },
        {
            "Scenario": "Invalid: high below low",
            "open": 100.00,
            "high": 98.00,
            "low": 102.00,
            "close": 100.00,
            "volume": 20000,
        },
        {
            "Scenario": "Invalid zero range",
            "open": 99.00,
            "high": 100.00,
            "low": 100.00,
            "close": 101.00,
            "volume": 20000,
        },
    ]

    df = pd.DataFrame(scenarios)

    df.insert(
        0,
        "datetime",
        pd.date_range(
            start="2026-01-02 04:00:00",
            periods=len(df),
            freq="h",
        ),
    )

    return df

def create_gap_test_scenarios() -> pd.DataFrame:
    """
    Creates valid OHLCV candles for one stock.

    Every row:
    - Has open, high, low, close, and volume
    - Satisfies valid OHLC relationships
    - Is chronologically ordered
    """

    scenarios = [
        {
            "Scenario": "First candle",
            "datetime": "2026-01-02 04:00:00",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 104.0,
            "volume": 10000,
        },
        {
            "Scenario": "Gap up",
            "datetime": "2026-01-02 08:00:00",
            "open": 106.0,       # Previous close = 104
            "high": 109.0,
            "low": 105.0,
            "close": 108.0,
            "volume": 12000,
        },
        {
            "Scenario": "No gap",
            "datetime": "2026-01-02 12:00:00",
            "open": 108.0,       # Previous close = 108
            "high": 110.0,
            "low": 107.0,
            "close": 109.0,
            "volume": 11000,
        },
        {
            "Scenario": "Gap down",
            "datetime": "2026-01-02 16:00:00",
            "open": 106.0,       # Previous close = 109
            "high": 107.0,
            "low": 103.0,
            "close": 104.0,
            "volume": 15000,
        },
        {
            "Scenario": "Tiny gap up",
            "datetime": "2026-01-02 20:00:00",
            "open": 104.0001,    # Previous close = 104
            "high": 105.0,
            "low": 103.5,
            "close": 104.5,
            "volume": 9000,
        },
        {
            "Scenario": "Tiny gap down",
            "datetime": "2026-01-03 00:00:00",
            "open": 104.4999,    # Previous close = 104.5
            "high": 105.0,
            "low": 103.0,
            "close": 103.5,
            "volume": 8500,
        },
        {
            "Scenario": "Large gap up",
            "datetime": "2026-01-03 04:00:00",
            "open": 115.0,       # Previous close = 103.5
            "high": 118.0,
            "low": 113.0,
            "close": 117.0,
            "volume": 30000,
        },
        {
            "Scenario": "Large gap down",
            "datetime": "2026-01-03 08:00:00",
            "open": 105.0,       # Previous close = 117
            "high": 107.0,
            "low": 102.0,
            "close": 103.0,
            "volume": 35000,
        },
        {
            "Scenario": "Gap up then bearish candle",
            "datetime": "2026-01-03 12:00:00",
            "open": 106.0,       # Previous close = 103
            "high": 107.0,
            "low": 101.0,
            "close": 102.0,
            "volume": 24000,
        },
        {
            "Scenario": "Gap down then bullish candle",
            "datetime": "2026-01-03 16:00:00",
            "open": 99.0,        # Previous close = 102
            "high": 104.0,
            "low": 98.0,
            "close": 103.0,
            "volume": 26000,
        },
        {
            "Scenario": "No gap with flat candle",
            "datetime": "2026-01-03 20:00:00",
            "open": 103.0,       # Previous close = 103
            "high": 103.0,
            "low": 103.0,
            "close": 103.0,
            "volume": 5000,
        },
        {
            "Scenario": "Gap up with flat candle",
            "datetime": "2026-01-04 00:00:00",
            "open": 105.0,       # Previous close = 103
            "high": 105.0,
            "low": 105.0,
            "close": 105.0,
            "volume": 4000,
        },
    ]

    df = pd.DataFrame(scenarios)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["volume"] = pd.to_numeric(df["volume"], errors="raise")

    return df.sort_values("datetime").reset_index(drop=True)




############# different candles 

def different_candles_logs():



    raw_data = [
        # ==========================================================
        # Scenario 1: Steady bullish movement — 10 candles
        # ==========================================================
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-01 00:00:00",
            "open": 100.00,
            "high": 101.50,
            "low": 99.60,
            "close": 101.00,
            "volume": 10000,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-01 04:00:00",
            "open": 101.10,
            "high": 102.40,
            "low": 100.80,
            "close": 102.00,
            "volume": 10500,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-01 08:00:00",
            "open": 102.00,
            "high": 103.70,
            "low": 101.70,
            "close": 103.20,
            "volume": 11000,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-01 12:00:00",
            "open": 103.10,
            "high": 104.60,
            "low": 102.90,
            "close": 104.10,
            "volume": 11500,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-01 16:00:00",
            "open": 104.20,
            "high": 105.90,
            "low": 103.90,
            "close": 105.40,
            "volume": 12200,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-01 20:00:00",
            "open": 105.30,
            "high": 106.80,
            "low": 105.00,
            "close": 106.20,
            "volume": 12800,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-02 00:00:00",
            "open": 106.10,
            "high": 107.70,
            "low": 105.80,
            "close": 107.30,
            "volume": 13500,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-02 04:00:00",
            "open": 107.40,
            "high": 109.00,
            "low": 107.10,
            "close": 108.60,
            "volume": 14200,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-02 08:00:00",
            "open": 108.50,
            "high": 110.10,
            "low": 108.20,
            "close": 109.70,
            "volume": 15000,
        },
        {
            "Scenario": "Steady bullish",
            "CandleShape": "Bullish",
            "datetime": "2026-01-02 12:00:00",
            "open": 109.60,
            "high": 111.30,
            "low": 109.30,
            "close": 110.90,
            "volume": 15800,
        },

        # ==========================================================
        # Scenario 2: Steady bearish movement — 10 candles
        # ==========================================================
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-02 16:00:00",
            "open": 111.00,
            "high": 111.30,
            "low": 109.50,
            "close": 109.80,
            "volume": 16500,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-02 20:00:00",
            "open": 109.90,
            "high": 110.20,
            "low": 108.30,
            "close": 108.70,
            "volume": 17000,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-03 00:00:00",
            "open": 108.60,
            "high": 109.00,
            "low": 107.20,
            "close": 107.50,
            "volume": 17600,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-03 04:00:00",
            "open": 107.60,
            "high": 107.90,
            "low": 106.10,
            "close": 106.40,
            "volume": 18100,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-03 08:00:00",
            "open": 106.50,
            "high": 106.80,
            "low": 104.90,
            "close": 105.20,
            "volume": 18800,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-03 12:00:00",
            "open": 105.30,
            "high": 105.70,
            "low": 103.80,
            "close": 104.10,
            "volume": 19400,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-03 16:00:00",
            "open": 104.20,
            "high": 104.50,
            "low": 102.60,
            "close": 102.90,
            "volume": 20100,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-03 20:00:00",
            "open": 103.00,
            "high": 103.30,
            "low": 101.40,
            "close": 101.70,
            "volume": 20900,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-04 00:00:00",
            "open": 101.80,
            "high": 102.10,
            "low": 100.20,
            "close": 100.50,
            "volume": 21700,
        },
        {
            "Scenario": "Steady bearish",
            "CandleShape": "Bearish",
            "datetime": "2026-01-04 04:00:00",
            "open": 100.60,
            "high": 100.90,
            "low": 98.90,
            "close": 99.20,
            "volume": 22500,
        },

        # ==========================================================
        # Scenario 3: Sideways, doji and spinning tops — 10 candles
        # ==========================================================
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Doji",
            "datetime": "2026-01-04 08:00:00",
            "open": 99.20,
            "high": 100.10,
            "low": 98.40,
            "close": 99.20,
            "volume": 14000,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Spinning top",
            "datetime": "2026-01-04 12:00:00",
            "open": 99.30,
            "high": 100.20,
            "low": 98.50,
            "close": 99.40,
            "volume": 13200,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Doji",
            "datetime": "2026-01-04 16:00:00",
            "open": 99.40,
            "high": 100.30,
            "low": 98.70,
            "close": 99.40,
            "volume": 12500,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Spinning top",
            "datetime": "2026-01-04 20:00:00",
            "open": 99.50,
            "high": 100.40,
            "low": 98.60,
            "close": 99.30,
            "volume": 12900,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Hammer",
            "datetime": "2026-01-05 00:00:00",
            "open": 99.40,
            "high": 99.70,
            "low": 96.80,
            "close": 99.60,
            "volume": 17500,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Hanging man",
            "datetime": "2026-01-05 04:00:00",
            "open": 99.50,
            "high": 99.90,
            "low": 96.90,
            "close": 99.30,
            "volume": 18000,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Bullish",
            "datetime": "2026-01-05 08:00:00",
            "open": 99.20,
            "high": 100.20,
            "low": 98.90,
            "close": 99.90,
            "volume": 15500,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Bearish",
            "datetime": "2026-01-05 12:00:00",
            "open": 100.00,
            "high": 100.30,
            "low": 99.00,
            "close": 99.40,
            "volume": 16000,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Flat",
            "datetime": "2026-01-05 16:00:00",
            "open": 99.40,
            "high": 99.40,
            "low": 99.40,
            "close": 99.40,
            "volume": 5000,
        },
        {
            "Scenario": "Sideways mixed candles",
            "CandleShape": "Doji",
            "datetime": "2026-01-05 20:00:00",
            "open": 99.40,
            "high": 100.00,
            "low": 98.80,
            "close": 99.40,
            "volume": 11000,
        },

        # ==========================================================
        # Scenario 4: Alternating gains and losses — 10 candles
        # ==========================================================
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bullish",
            "datetime": "2026-01-06 00:00:00",
            "open": 99.50,
            "high": 102.00,
            "low": 99.20,
            "close": 101.50,
            "volume": 21000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bearish",
            "datetime": "2026-01-06 04:00:00",
            "open": 101.60,
            "high": 102.00,
            "low": 98.90,
            "close": 99.30,
            "volume": 22000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bullish",
            "datetime": "2026-01-06 08:00:00",
            "open": 99.40,
            "high": 102.40,
            "low": 99.10,
            "close": 102.00,
            "volume": 23000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bearish",
            "datetime": "2026-01-06 12:00:00",
            "open": 102.10,
            "high": 102.50,
            "low": 99.40,
            "close": 99.80,
            "volume": 24000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bullish",
            "datetime": "2026-01-06 16:00:00",
            "open": 99.70,
            "high": 103.00,
            "low": 99.30,
            "close": 102.60,
            "volume": 25000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bearish",
            "datetime": "2026-01-06 20:00:00",
            "open": 102.70,
            "high": 103.10,
            "low": 99.80,
            "close": 100.20,
            "volume": 26000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bullish",
            "datetime": "2026-01-07 00:00:00",
            "open": 100.30,
            "high": 103.60,
            "low": 99.90,
            "close": 103.20,
            "volume": 27000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bearish",
            "datetime": "2026-01-07 04:00:00",
            "open": 103.30,
            "high": 103.80,
            "low": 100.00,
            "close": 100.50,
            "volume": 28000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bullish",
            "datetime": "2026-01-07 08:00:00",
            "open": 100.40,
            "high": 104.00,
            "low": 100.10,
            "close": 103.50,
            "volume": 29000,
        },
        {
            "Scenario": "Alternating movement",
            "CandleShape": "Bearish",
            "datetime": "2026-01-07 12:00:00",
            "open": 103.60,
            "high": 104.10,
            "low": 100.20,
            "close": 100.70,
            "volume": 30000,
        },

        # ==========================================================
        # Scenario 5: Strong upward movement and reversal — 10 candles
        # ==========================================================
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Bullish marubozu",
            "datetime": "2026-01-07 16:00:00",
            "open": 100.70,
            "high": 104.00,
            "low": 100.70,
            "close": 104.00,
            "volume": 35000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Bullish marubozu",
            "datetime": "2026-01-07 20:00:00",
            "open": 104.00,
            "high": 108.00,
            "low": 104.00,
            "close": 108.00,
            "volume": 42000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Bullish",
            "datetime": "2026-01-08 00:00:00",
            "open": 108.20,
            "high": 112.50,
            "low": 107.80,
            "close": 112.00,
            "volume": 50000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Bullish",
            "datetime": "2026-01-08 04:00:00",
            "open": 112.10,
            "high": 117.00,
            "low": 111.70,
            "close": 116.50,
            "volume": 58000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Hanging man",
            "datetime": "2026-01-08 08:00:00",
            "open": 116.70,
            "high": 117.40,
            "low": 111.00,
            "close": 116.20,
            "volume": 65000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Doji",
            "datetime": "2026-01-08 12:00:00",
            "open": 116.20,
            "high": 118.00,
            "low": 114.50,
            "close": 116.20,
            "volume": 62000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Bearish",
            "datetime": "2026-01-08 16:00:00",
            "open": 116.00,
            "high": 116.50,
            "low": 111.50,
            "close": 112.20,
            "volume": 72000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Bearish marubozu",
            "datetime": "2026-01-08 20:00:00",
            "open": 112.20,
            "high": 112.20,
            "low": 107.50,
            "close": 107.50,
            "volume": 81000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Bearish",
            "datetime": "2026-01-09 00:00:00",
            "open": 107.70,
            "high": 108.20,
            "low": 103.00,
            "close": 103.50,
            "volume": 76000,
        },
        {
            "Scenario": "Bullish surge and reversal",
            "CandleShape": "Hammer",
            "datetime": "2026-01-09 04:00:00",
            "open": 103.30,
            "high": 104.50,
            "low": 97.00,
            "close": 104.00,
            "volume": 83000,
        },

        # ==========================================================
        # Scenario 6: Large crash and recovery — 10 candles
        # ==========================================================
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Bearish",
            "datetime": "2026-01-09 08:00:00",
            "open": 103.80,
            "high": 104.20,
            "low": 98.00,
            "close": 98.50,
            "volume": 90000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Bearish marubozu",
            "datetime": "2026-01-09 12:00:00",
            "open": 98.50,
            "high": 98.50,
            "low": 88.00,
            "close": 88.00,
            "volume": 150000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Bearish",
            "datetime": "2026-01-09 16:00:00",
            "open": 88.20,
            "high": 89.00,
            "low": 80.00,
            "close": 81.50,
            "volume": 190000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Hammer",
            "datetime": "2026-01-09 20:00:00",
            "open": 81.00,
            "high": 84.00,
            "low": 72.00,
            "close": 83.00,
            "volume": 220000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Doji",
            "datetime": "2026-01-10 00:00:00",
            "open": 83.00,
            "high": 86.00,
            "low": 80.00,
            "close": 83.00,
            "volume": 180000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Bullish",
            "datetime": "2026-01-10 04:00:00",
            "open": 83.20,
            "high": 88.00,
            "low": 82.50,
            "close": 87.00,
            "volume": 160000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Bullish marubozu",
            "datetime": "2026-01-10 08:00:00",
            "open": 87.00,
            "high": 92.00,
            "low": 87.00,
            "close": 92.00,
            "volume": 145000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Bullish",
            "datetime": "2026-01-10 12:00:00",
            "open": 92.10,
            "high": 96.50,
            "low": 91.50,
            "close": 95.50,
            "volume": 130000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Spinning top",
            "datetime": "2026-01-10 16:00:00",
            "open": 95.60,
            "high": 98.00,
            "low": 93.50,
            "close": 95.80,
            "volume": 105000,
        },
        {
            "Scenario": "Crash and recovery",
            "CandleShape": "Bullish",
            "datetime": "2026-01-10 20:00:00",
            "open": 95.70,
            "high": 100.00,
            "low": 95.00,
            "close": 99.20,
            "volume": 98000,
        },
    ]


    df = pd.DataFrame(raw_data)

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )
    return df

def get_fake_data():
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    df = pd.read_csv(f"{script_dir}/fake_data.csv",parse_dates=["datetime"],)
    return df
