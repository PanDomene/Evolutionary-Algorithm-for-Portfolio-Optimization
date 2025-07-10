import numpy as np
import pandas as pd
from . import utils as u

def MBF(algorithm, data, runs=10, generations=50, **kwargs):
    """
    Calculate and print the Mean Best Fitness (MBF) over multiple runs.

    Args:
    - algorithm (class): Evolutionary algorithm class.
    - data (pd.DataFrame): Historical closing prices.
    - runs (int, optional): Number of runs (default is 10).
    - generations (int, optional): Num. of generations per run (default is 50).
    - **kwargs: Additional keyword arguments for the algorithm.

    Returns:
    - best_fitness (np.ndarray): Array of best fitness values from each run.
    - returns (np.ndarray): Array of best expected returns obtained on the last
      generation of each run.
    - risks (np.ndarray): Array with the risks associated to the best individual
      in the las generation for each run.
    """
    best_fitness = np.empty(runs)
    returns = np.empty(runs)
    risks = np.empty(runs)
    for i in range(runs):
        ea = algorithm(data, **kwargs)
        ea.run(generations)
        best = ea.population[np.argmax(ea.pop_fitness)][:-1]
        f, r, risk = ea.fitness(best)
        best_fitness[i] = f
        returns[i] = r
        risks[i] = risk
    print('MBF: ', best_fitness.mean())
    return best_fitness, returns, risks



def AES_SR(algorithm, data, solution=0.26, max_risk=18,
           runs=10, max_gens=50, **kwargs):
    """
    Calculate the Average Fitness Evaluations to Solution (AES) and the Success
    Rate (SR) over a series of runs.

    Args:
    - algorithm (class): Evolutionary algorithm class.
    - data (pd.DataFrame): Historical closing prices.
    - solution (float, optional): Target solution value (default is 0.26).
    - max_risk (float, optional): Maximum risk (expressed as a %) allowed for 
      the portfolio to be considered a success (default is 18%).
    - runs (int, optional): Number of runs (default is 10).
    - max_gens (int, optional): Maximum number of generations to run. If the
      solution is not found within this number of generations, that run's
      evaluations are not included in the calculation of AES. (default is 100).
    - **kwargs: Additional keyword arguments for the algorithm.

    Returns:
    - evaluations (np.ndarray): Array of fitness evaluations required to reach
      the solution in each run. Runs where no solution was found have a nan value.
    - returns (np.ndarray): Array of best expected returns obtained on the last
      generation of each run.
    - risks (np.ndarray): Array with the risks associated to the best individual
      if the las generation for each run.
    """
    evaluations = np.empty(runs)
    returns = np.empty(runs)
    risks = np.empty(runs)
    fails = 0
    for i in range(runs):
        ea = algorithm(data, **kwargs)
        gen = 0
        best = ea.population[np.argmax(ea.pop_fitness)][:-1]
        _, r, risk = ea.fitness(best)
        while r < solution or risk > max_risk: # while no solution is found
            ea.run(1)
            best = ea.population[np.argmax(ea.pop_fitness)][:-1]
            _, r, risk = ea.fitness(best)
            gen += 1
            if gen == max_gens: # If no solution found in max_gens generations
                ea.fitness_evaluations = np.nan # don't count this run.
                fails += 1 # Keep trak of failures of success.
                break
        returns[i] = r
        risks[i] = risk
        evaluations[i] = ea.fitness_evaluations
    successes = runs - fails
    print('AES: ', np.nanmean(evaluations))
    print('SR: ', successes/runs)
    return evaluations, returns, risks



def asset_robustness(algorithm, data, sample_size=100,
                     runs=5, generations=50, **kw):
    """
    Helps evaluate the robustness of the evolutionary algorithm to different
    groups of assets, i.e. to different problem instances.

    Args:
    - algorithm (class, optional): Evolutionary algorithm class.
    - data (pd.DataFrame, optional): All asset closing data.
    - frac (float, optional): The percentage of all assets to use in each run
      (default is 0.1).
    - runs (int, optional): Number of runs (default is 5).
    - generations (int, optional): Number of generations per run
      (default is 100).
    - **kw: Additional keyword arguments for the algorithm.

    Returns:
    - np.ndarray: Array of best fitness values from each run.
    """
    fits = np.empty(runs)
    for i in range(runs):
        print('run: ', i+1)
        data_i = data.sample(sample_size, axis=1)
        ea = algorithm(data_i, **kw)
        ea.run(generations)
        ea.plot_max_fitness()
        fits[i] = ea.best_fitness
    return fits



def rank_data_(data, size):
    """
    Selects the top `size` assets based on risk-adjusted returns.

    Computes annualized returns and standard deviation for each asset,
    ranks them by return-to-risk ratio, and returns the top `size` assets.

    Parameters:
    ----------
    data : pandas.DataFrame
        Asset price time series, one asset per column.
    size : int
        Number of top-ranked assets to return.

    Returns:
    -------
    pandas.DataFrame
        Subset of `data` with the top `size` ranked assets.
    """
    daily_returns = data.pct_change().mean()*252
    risks = data.std()
    ranking = daily_returns/risks
    ranked = ranking.sort_values(ascending=False).index
    ranked_subset = ranked[:size]
    return data[ranked_subset]



def time_robustness(algorithm, data, periods=5,
                    generations=50, n_assets=100, **kw):
    """
    Helps evaluate the robustness of the evolutionary algorithm to different
    time periods, given a fixed group of assets. This situation resembles a
    real-life situation, where every day the data is slightly different from the
    previous one and, over a long period of time, it can vary significantly.

    Args:
    - algorithm (class, optional): Evolutionary algorithm class.
    - data (pd.DataFrame, optional): All asset closing data.
    - periods (int, optional): The number of time periods the data will be slit
      on (default is 5).
    - generations (int, optional): Number of generations per run
      (default is 100).
    - **kw: Additional keyword arguments for the algorithm.

    Returns:
    - np.ndarray: Array of best fitness values from each run.
    """
    fits = np.empty(periods)
    data = np.array_split(data, periods)
    for i, period_data in enumerate(data):
        period = rank_data_(period_data, n_assets)
        print(f'period #{i+1}')
        start = str(period.index[0])
        end = str(period.index[-1])
        print('start: ', start)
        print('end: ', end)
        ea = algorithm(period, **kw)
        ea.run(generations)
        ea.plot_max_fitness(label=f'{start[5:7]}/{start[:4]}-{end[5:7]}/{end[:4]}')
        fits[i] = ea.best_fitness
        print('best fitness: ', fits[i])
        print('------------------------')
    plt.legend(bbox_to_anchor=(1.0, 1.02))
    return fits


def returns_to_risk_ratio(ticks, n, display=False):
    """
    Selects the top N assets with the highest return-to-risk ratio.

    Parameters:
    - ticks (list): List of asset tickers to consider.
    - n (int): Number of top assets to select.
    - display (bool): If True, prints the resulting DataFrame (default: False).

    Returns:
    - numpy.ndarray: Tickers of the top N assets.
    - pandas.DataFrame: Return-to-risk ratios and annual returns of the top N assets.
    """
    national = u.all_data[ticks]
    daily_returns = national.pct_change().dropna()
    anual_returns = daily_returns.mean() * 252
    risks = daily_returns.std()
    top_n = (anual_returns/risks).sort_values(ascending=False)[:n]
    top_n_returns = anual_returns[top_n.index]
    top = {"return/risk": top_n, "anual return": top_n_returns}
    top = pd.DataFrame(top)
    if display:
        print(top)
    
    return top_n.index.values, top
