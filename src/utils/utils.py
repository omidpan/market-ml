# filename: utils.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# import yfinance as yf

from config import (MODEL_PATH, SCALER_PATH,
    FEATURE_META_PATH, RANDOM_SEED, TRAIN_FRAC, VAL_FRAC
)


def set_seeds(seed=RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


# def download_raw_data(ticker, period, interval) -> pd.DataFrame:
#     """Downloads raw market history from Yahoo Finance and formats indices."""
#     df = yf.download(ticker, period=period, interval=interval, prepost=True)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.droplevel('Ticker')
#     df.index = df.index.tz_localize(None)
#     return df


def chronological_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """
    Splits a time-ordered dataframe into train/val/test by position, never
    shuffling. Fit any scaler ONLY on the train slice this returns -- fitting
    on the full dataset first (as the original code did) leaks test-period
    statistics into training.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def create_sequences(feature_df: pd.DataFrame, target_series: pd.Series, window_size: int):
    """
    Builds rolling windows X (window_size x n_features) and matching targets Y.
    X[i] uses rows [i : i+window_size) of features; Y[i] is the target value
    aligned to the LAST row in that window (row i+window_size-1) -- i.e. the
    return already computed from that candle to `horizon` candles later.
    No future rows are read past that point.
    """
    X, Y = [], []
    feat_vals = feature_df.values
    targ_vals = target_series.values
    for i in range(len(feat_vals) - window_size + 1):
        X.append(feat_vals[i:i + window_size, :])
        Y.append(targ_vals[i + window_size - 1])
    return np.array(X), np.array(Y)


def load_model_and_scaler(task:str,bar_size:str):
    """Loads the trained Keras model, scaler, and feature metadata from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler files are missing. Please run train_lstm.py first.")
    model = keras.models.load_model(f'{MODEL_PATH}-{task}-{bar_size}.keras')
    scaler = joblib.load(f'{SCALER_PATH}-{task}-{bar_size}.pkl')
    with open(FEATURE_META_PATH) as f:
        meta = json.load(f)
    return model, scaler, meta
