import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .algorithms import random_search, EA

# The expected returns for the portfolio
def expected_returns(chromosome, returns):
    return chromosome @ returns

# And the risk associated, expressed as a %
def get_risk(chromosome, cov_matrix):
    return np.sqrt(252 * chromosome @ cov_matrix @ chromosome)


def evaluate_out_of_sample(weights, test_data, plot=True, verbose=True):
    """
    Evaluates the performance of a fixed portfolio on out-of-sample data.

    Args:
    - weights (np.ndarray): Portfolio weights (1D array).
    - test_data (pd.DataFrame): Closing prices for the test period (columns = assets).
    - plot (bool): Whether to plot cumulative return.
    - verbose (bool): Whether to print realized return, expected return, and volatility.

    Returns:
    - actual_return (float): Realized return over the test period (in percent).
    - risk (float): Annualized volatility (in percent).
    """
    # Daily arithmetic returns
    daily_returns = test_data.pct_change().dropna()
    portfolio_returns = daily_returns @ weights

    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Realized metrics
    actual_return = cumulative_returns.iloc[-1]
    risk = portfolio_returns.std() * np.sqrt(252)

    if plot:
        plt.figure(figsize=(7, 3))
        plt.plot(cumulative_returns, label='Out-of-sample cumulative return')
        plt.title('Out-of-Sample Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if verbose:
        print(f"Actual return: {round(100 * actual_return, 2)}%")
        print(f"Volatility during test: {round(100 * risk, 2)}%")

    return actual_return, risk


def test_portfolio_weights(weights, train_data, test_data, plot=True, verbose=True):
    """
    Evaluate the performance of a given portfolio.

    Args:
        weights (np.ndarray or pd.Series): Portfolio weights.
        train_data (pd.DataFrame): Training closing prices.
        test_data (pd.DataFrame): Testing closing prices.
        plot (bool): Whether to plot the out-of-sample performance.
        verbose (bool): Whether to print training metrics.

    Returns:
        Tuple of actual return and risk from evaluate_out_of_sample.
    """
    daily_returns_train = train_data.pct_change().dropna().values
    expected_daily_returns = np.mean(daily_returns_train, axis=0)
    cov_matrix = np.cov(daily_returns_train.T)

    train_return = expected_daily_returns @ weights * len(test_data)
    train_risk = np.sqrt(252 * weights @ cov_matrix @ weights)

    if verbose:
        print(f"Expected return (train): {round(100 * train_return, 3)}%")
        print(f"Risk (train): {round(100 * train_risk, 3)}%")

    return evaluate_out_of_sample(weights, test_data, plot, verbose)