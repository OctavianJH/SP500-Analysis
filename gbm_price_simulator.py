import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_prices, compute_log_returns

prices = load_prices("data/sp500_prices.csv")
returns = compute_log_returns(prices)

mu = returns.mean()
sigma = returns.std()

def sim_gbm(S0, mu, sigma, T, N):
    dt = T / N
    prices = np.zeros(N)
    prices[0] = S0

    for t in range(1, N):
        Z = np.random.normal()
        prices[t] = prices[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return prices

S0 = prices.iloc[-1]   # last observed price
T = 1.0                # 1 year
N = 252                # trading days

path = sim_gbm(S0, mu, sigma, T, N)

num_simulations = 10_000
terminal_prices = []

for _ in range(num_simulations):
    path = sim_gbm(S0, mu, sigma, T, N)
    terminal_prices.append(path[-1])

terminal_prices = np.array(terminal_prices)

plt.hist(terminal_prices, bins=50, density=True)
plt.title("Distribution of Simulated S&P 500 Prices at T = 1 Year")
plt.xlabel("Price")
plt.ylabel("Density")
plt.show()

log_terminal_prices = np.log(terminal_prices)

plt.hist(log_terminal_prices, bins=50, density=True)
plt.title("Log of Terminal Prices")
plt.xlabel("log(price)")
plt.ylabel("Density")
plt.show()

import matplotlib.pyplot as plt
from scipy.stats import norm

mu_log = log_terminal_prices.mean()
sigma_log = log_terminal_prices.std()
x = np.linspace(min(log_terminal_prices), max(log_terminal_prices), 1000)
plt.hist(log_terminal_prices, bins=50, density=True, alpha=0.6, label="Simulated")
plt.plot(x, norm.pdf(x, mu_log, sigma_log), 'r', label="Theoretical normal")
plt.title("Log-terminal price vs theoretical normal")
plt.legend()
plt.show()

DEBUG_SINGLE_PATH = False

if DEBUG_SINGLE_PATH:
    path = sim_gbm(S0, mu, sigma, T, N)
    plt.plot(path)
    plt.title("Single GBM Price Path")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.show()