## **Monte Carlo Simulation of the S&P 500 Using Geometric Brownian Motion**

## Overview:
This project implements a Monte Carlo Simulation of the S&P 500's price dynamics under a Geometric Brownian Motion model. Using historical data from 1st January 2015 to the Present, the model estimates drift and volatility parameters, simulates future price paths and validates numerical results against the analytical GBM Solution.

## Results:
The simulation reproduces the theoretical mean and variance of the GBM terminal price distribution with relative errors below 2%.

This agreement demonstrates:
1. Correct discretisation of the GBM stochastic differential equation
2. Proper scaling of drift and volatility parameters
3. Convergence of Monte Carlo estimates to analytical results

## Key Limitations:
The GBM framework relies on strong assumptions, including constant volatility, normally distributed log returns, and the absence of jumps or extreme market events. These assumptions limit the model’s realism, particularly during periods of market stress.

## Future Work:
- Confidence interval estimation and value-at-risk style analysis
- Comparison with alternative stochastic models (e.g. stochastic volatility, jump diffusion)
- Stress-testing under crisis conditions

## Author:
Octavian Humphreys - Independant Quantitative finance project