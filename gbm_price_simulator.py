import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_prices, compute_log_returns

n_steps = 252        # trading days
n_paths = 10_000     # mc simulations

prices = load_prices("data/sp500_prices.csv")
returns = compute_log_returns(prices)

mu = returns.mean()
sigma = returns.std()

mu_annual = mu * 252
sigma_annual = sigma * np.sqrt(252)

def sim_gbm(S0, mu, sigma, T, n_steps):
    dt = T / n_steps
    prices = np.zeros(n_steps)
    prices[0] = S0

    for t in range(1, n_steps):
        Z = np.random.normal()
        prices[t] = prices[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    return prices


S0 = prices.iloc[-1]   # last observed price
T = 1.0                # 1 year
N = 252                # trading days

num_simulations = 10_000
terminal_prices = []

for _ in range(n_paths):
    path = sim_gbm(S0, mu_annual, sigma_annual, T, n_steps)
    terminal_prices.append(path[-1])

terminal_prices = np.array(terminal_prices)

mc_mean = terminal_prices.mean()
mc_var  = terminal_prices.var()

theory_mean = S0 * np.exp(mu_annual * T)
theory_var = (S0**2) * np.exp(2 * mu_annual * T) * (np.exp(sigma_annual**2 * T) - 1)

mean_error = abs(mc_mean - theory_mean) / theory_mean
var_error = abs(mc_var - theory_var) / theory_var

print("Mean error:", mean_error)
print("Var error :", var_error)



# --- PLOTS to enable after validation ---

plt.hist(terminal_prices, bins=50, density=True)
plt.title("Distribution of Simulated S&P 500 Prices at T = 1 Year")
plt.xlabel("Price")
plt.ylabel("Density")
plt.show()

log_terminal_prices = np.log(terminal_prices / S0)

plt.hist(log_terminal_prices, bins=50, density=True)
plt.title("Log of Terminal Prices")
plt.xlabel("log(price / S0)")
plt.ylabel("Density")
plt.show()

