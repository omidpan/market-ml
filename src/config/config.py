# filename: config.py
import os
from pathlib import Path
# Kafka Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
RAW_DATA_TOPIC = "ionq_raw_candles"
PREDICTION_TOPIC = "ionq_predictions"

# Ticker & Modeling Settings
WINDOW_SIZE = 15 # 3 trading weeks
RANDOM_SEED = 2505
HORIZON = 1 # predict return this many candles ahead
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining fraction (0.15) is TEST
# File Paths
MODEL_PATH = BASE_DIR / 'docker' /'models' / 'model' / 'lstm_model'
SCALER_PATH = BASE_DIR / 'docker' /'models' / 'model' / 'scaler'
FEATURE_META_PATH = BASE_DIR / 'docker' /'models' / 'feature_meta'
DATA_DIR = BASE_DIR / 'data'
# Strategy Parameters
CONFIDENCE_THRESHOLD = 0.005      # 0.5% minimum expected return to trigger trade
NO_TRADE_ZONE_LOWER = -0.002
NO_TRADE_ZONE_UPPER = 0.002
STOP_LOSS_PCT = 0.015             # 1.5% Stop Loss
TAKE_PROFIT_PCT = 0.03            # 3.0% Take Profit
BASE_POSITION_SIZE = 100          # Base units to trade
print(f"System locked to Base Directory: {BASE_DIR}")