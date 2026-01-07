import pandas as pd
import numpy as np

def load_prices(csv_path):
    df = pd.read_csv(csv_path)
    return df["Close"]

def compute_log_returns (prices):
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()
