import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/sp500_prices.csv")
#Formatting
prices = df["Close"]
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

print("Mean return", returns.mean())
print("Voltality (std):", returns.std())

#Plotting data as histogram
plt.hist(returns, bins=50)
plt.title("Histogram of S&P 500 Daily Log Returns")
plt.xlabel("Log return")
plt.ylabel("Frequency")
plt.show()

print(df.head(5))   # first 5 rows
print(df.tail(5))