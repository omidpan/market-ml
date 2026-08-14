# filename: predict_realtime.py
import json
import time
import numpy as np
import pandas as pd
# pip install kafka-python
from kafka import KafkaConsumer, KafkaProducer

from config import TICKER ,INTERVAL, WINDOW_SIZE, MODEL_PATH, SCALER_PATH, FEATURE_META_PATH
from utils.utils import download_raw_data, load_model_and_scaler
from feature_engineering import add_technical_indicators
import argparse
from config import DATA_DIR
parser = argparse.ArgumentParser(description="Process a stock symbol.")
parser.add_argument("-bs", "--bar_size",
                    type=str,
                    required=True,
                    help="candle size of historical data. default is 1 hour.", 
                    nargs='?', default="1 hour")
parser.add_argument(
    "-t ","--task",
    choices=["classification", "regression", "multi_regression"],
    default="classification",
    help="Model type to train (default: regression).",
)
parser.add_argument(
    "-s ","--sector",
    choices=["quantum", "semiconductor"],
    default="semiconductor",
    help="Model type to train (default: regression).",
)
args = parser.parse_args()

# Access the value using dot notation
stock_symbol = args.symbol.lower()
bar_size = args.bar_size
task=args.task
sector=args.sector
KAFKA_SERVER = 'localhost:9092'
RAW_DATA_TOPIC = 'ionq_raw_candles'
PREDICTION_TOPIC = 'ionq_predictions'

# Longest rolling lookback used in add_technical_indicators is 26 (EMA_26).
# Keep extra buffer above WINDOW_SIZE so recomputed indicators are valid for
# every row inside the model's input window -- otherwise the first several
# rows of the window would carry NaN-derived indicators that training never saw.
INDICATOR_LOOKBACK = 0

def fetch_initial_buffer(sector:str,buffer_size):
    print(f"Fetching initial tail data from {sector} sector candle buffer...")
    df = pd.read_csv(f"{DATA_DIR}/{sector}/dataset_{bar_size}.csv")
    return df.tail(buffer_size)


def main():
    model, scaler, meta = load_model_and_scaler(task=task,bar_size=bar_size)
    feature_columns = meta['feature_columns']
    window_size = meta['window_size']
    buffer_size = window_size + INDICATOR_LOOKBACK

    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    consumer = KafkaConsumer(
        RAW_DATA_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='ionq_predictor',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    raw_buffer = fetch_initial_buffer(sector,buffer_size)
    print(f"Real-time prediction engine started for {sector} with buffer_size:{buffer_size} Listening for completed candles...")

    for message in consumer:
        print(f"Received candle: {message.value}")
        candle = message.value
        ts = pd.to_datetime(candle.get('datetime', time.time()), unit='s')
        new_row = pd.DataFrame([{
            'open': candle['open'], 'high': candle['high'],
            'low': candle['low'], 'close': candle['close'],
            'volume': candle['volume']
        }], index=[ts])

        raw_buffer = pd.concat([raw_buffer, new_row]).iloc[-buffer_size:]
        if len(raw_buffer) < buffer_size:
            continue  # not enough history yet for indicators + full window

        # Recompute indicators the SAME way training does, so live features
        # never silently diverge from what the model was trained on.
        feat_df = add_feature(raw_buffer)
        if len(feat_df) < window_size:
            print("Not enough rows after adding indicators. Waiting for more data...")
            continue

        window_feats = feat_df[feature_columns].values[-window_size:]
        scaled_window = scaler.transform(window_feats)
        X_input = np.expand_dims(scaled_window, axis=0)  # (1, window_size, n_features)

        predicted_return = float(model.predict(X_input, verbose=0)[0][0])

        # The window's last row IS the latest completed candle -- this is
        # "current" price, not iloc[-2]. Using -2 (as the original code did)
        # silently shifts every predicted return by one candle.
        current_close = float(feat_df['close'].iloc[-1])
        predicted_close = current_close * (1 + predicted_return)

        payload = {
            "timestamp": time.time(),
            "ticker": TICKER,
            "current_close": current_close,
            "predicted_close": predicted_close,
            "predicted_return": predicted_return
        }

        producer.send(PREDICTION_TOPIC, value=payload)
        print(f"Published Prediction -> Return: {predicted_return*100:.2f}% | "
              f"Current: {current_close:.2f} | Pred: {predicted_close:.2f}")


if __name__ == '__main__':
    main()
