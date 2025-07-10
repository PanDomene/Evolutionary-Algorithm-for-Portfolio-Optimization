import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .algorithms import random_search, EA

# The expected returns for the portfolio
def expected_returns(chromosome, returns):
    return chromosome @ returns

# And the risk associated, expressed as a %
def get_risk(chromosome, cov_matrix):
    return 100 * np.sqrt(252 * chromosome @ cov_matrix @ chromosome)

def split_data(data):
    """
    Splits price data into training (70%) and testing (30%) sets, then computes:

    - Annualized returns from training daily returns
    - Daily returns for the test set
    - Covariance matrix from training daily returns

    Parameters
    ----------
    data (pandas.DataFrame): Historical price data with assets as columns.

    Returns
    -------
    - train_annual_returns (pandas.Series): Annualized returns per asset from the training set.
    - test_daily_returns (pandas.DataFrame): Daily returns per asset from the test set.
    - covariance_matrix (pandas.DataFrame): Covariance matrix of training daily returns.
    """
    train = data.iloc[:int(len(data)*0.7)] # to evolve and select the portfolio
    test = data.iloc[int(len(data)*0.7):] # to evaluate how the selected portfolio performs on unseen data

    train_daily_returns = train.pct_change().dropna()
    train_annual_returns = train_daily_returns.mean() * 252

    test_daily_returns = test.pct_change().dropna()
    
    covariance_matrix = train_daily_returns.cov()
    
    return train_annual_returns, test_daily_returns, covariance_matrix



def evaluate_out_of_sample(weights, test_returns, plot=True):
    """
    Evaluates the performance of a fixed portfolio on out-of-sample data.

    Args:
    - weights (np.ndarray): Portfolio weights (1D array).
    - test_prices (pd.DataFrame): Price data for the test period (columns = assets).
    - plot (bool): Whether to plot cumulative return.
    """
    
    # Daily portfolio returns
    portfolio_returns = test_returns @ weights

    # Cumulative returns over time
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Annualized expected return and risk
    expected_return = 100 * portfolio_returns.mean() * 252
    risk = 100 * portfolio_returns.std() * np.sqrt(252)

    # Actual return over the entire test period
    actual_return = 100 * cumulative_returns.iloc[-1]

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

    print(f'Actual return was {round(actual_return, 2)}% with a volatility of {round(risk, 2)}%')



def test_random(closing_prices):
    train, test, cov_matrix = split_data(closing_prices)

    population = random_search(cov_matrix)
    
    fitness_vals = np.zeros(len(population))
    risks = np.zeros(len(population))
    
    for i, chromosome in enumerate(population):
        fitness_vals[i] = expected_returns(chromosome, train)
        risks[i] = get_risk(chromosome, cov_matrix)
    
    best = population[np.argmax(fitness_vals)]

    print(f'mean training returns: {round(100*fitness_vals.mean(), 3)}%')
    print(f'mean training risk: {round(risks.mean(), 3)}%')
    print(f'best training returns in population: {round(100*fitness_vals.max(), 3)}%')
    print(f'with risk: {round(risks[np.argmax(fitness_vals)], 3)}%')

    evaluate_out_of_sample(best, test)


def test_EA(closing_prices, generations=50, **kwargs):
    n = len(closing_prices)
    train = closing_prices[:int(0.7 * n)]
    test = closing_prices[int(0.7 * n):]
    test = test.pct_change().dropna()
    
    ea = EA(train, **kwargs)
    ea.run(generations)

    x = np.argmax(ea.pop_fitness)
    best = ea.population[x,:-1]
    _, returns, risk = ea.fitness(best)

    print(f'mean training returns: {round(100*ea.pop_fitness.mean(), 3)}%')
    print(f'mean training risk: {round(np.mean([100*ea.volatility(chrom) for chrom in
                                            ea.population]), 3)}%')
    print(f'best training returns in population: {round(100*returns, 3)}%')
    print(f'with risk: {round(risk, 3)}%')

    evaluate_out_of_sample(best, test)