import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import data as dta
from .algorithms import EA

def MBF(train_data, test_data, runs=10, generations=50, **kwargs):
    """
    Evaluate the Mean Best Fitness (MBF) on the test set across multiple runs.

    Args:
    - train_data (pd.DataFrame): Closing prices of train data.
    - test_data (pd.DataFrame): Closing prices of test data.
    - runs (int): Number of independent runs.
    - generations (int): Number of generations per run.
    - **kwargs: Extra args passed to the evolutionary algorithm.

    Returns:
    - actual_returns (np.ndarray): Out-of-sample total returns from each run.
    - risks (np.ndarray): Estimated risk based on the training-set covariance matrix.
    """
    
    actual_returns = np.empty(runs)
    risks = np.empty(runs)
    
    for i in range(runs):
        ea = EA(train_data, **kwargs)
        ea.run(generations)

        # Evaluate best individual on the test set
        actual_return, _ = ea.test_returns(test_data, ea.best_chrom)
        risk = ea.volatility(ea.best_chrom)

        actual_returns[i] = actual_return
        risks[i] = risk

    print(f'Mean Return at the end of testing period: {round(100 * actual_returns.mean(), 2)}%')
    print(f'Mean Risk: {round(100 * risks.mean(), 2)}%')

    return actual_returns, risks



def AES_SR(train_data, test_data, solution=(14, 16), runs=10, 
           max_gens=50, plot=False, **kwargs):
    """
    Calculate the Average Fitness Evaluations to Solution (AES) and the Success
    Rate (SR) using out-of-sample returns and training-based risk.

    Args:
    - train_data (pd.DataFrame): Historical closing prices of train data (used for evolution and risk estimation).
    - test_data (pd.DataFrame): Historical closing prices of test data (used for return evaluation).
    - solution (tuple): Minimum actual return (%) on test data and maximum risk (%) on train data to consider the run a success.
    - max_risk (float): Maximum risk (estimated from training cov matrix) allowed (%).
    - runs (int): Number of independent runs.
    - max_gens (int): Maximum number of generations per run.
    - plot (bool): If True, creates a scatterplot of risk vs. returns, highlighting the successes.
    - **kwargs: Extra arguments for the algorithm.

    Returns:
    - evaluations (np.ndarray): Fitness evaluations per run (NaN if failed).
    - returns (np.ndarray): Actual out-of-sample return per run.
    - risks (np.ndarray): Risk per run (based on training cov matrix).
    """

    evaluations = np.empty(runs)
    returns = np.empty(runs)
    risks = np.empty(runs)
    fails = 0

    actual_return, estimated_risk = np.nan, np.nan  # safe defaults

    for i in range(runs):
        ea = EA(train_data, **kwargs)
        gen = 0

        while gen < max_gens:
            ea.run(1)
            
            # Evaluate return on test set
            actual_return, _  = ea.test_returns(test_data, ea.best_chrom) 
            # Evaluate risk on training set
            estimated_risk = ea.volatility(ea.best_chrom)
            
            if actual_return >= solution[0] and estimated_risk <= solution[1]:
                break  # Found valid solution

            gen += 1

        if gen == max_gens:  # Did not meet criteria
            evaluations[i] = np.nan
            fails += 1
        else:
            evaluations[i] = ea.fitness_evaluations

        # Record values regardless of success
        returns[i] = actual_return
        risks[i] = estimated_risk

    successes = runs - fails
    print(f'AES: {np.nanmean(evaluations):.2f}')
    print(f'SR: {successes / runs:.2f}')

    if plot:
        c1 = "red"
        c2 = "steelblue"
        colors = [c1 if (ret > solution[0] and rsk < solution[1]) 
                  else c2 for ret, rsk in zip(returns, risks)]
        
        plt.figure(figsize=(5, 5))
        plt.scatter(risks, returns, c=colors)
        plt.xlabel("Risk (%)")
        plt.ylabel("Return (%)")
        
        min_risk, max_risk = risks.min(), risks.max()
        min_return, max_return = returns.min(), returns.max()
        
        # Add small margins so lines and points aren't right on the edges
        x_margin = 0.05 * (max_risk - min_risk)
        y_margin = 0.05 * (max_return - min_return)
        
        # Adaptively set axis limits
        x_min = min_risk - x_margin
        x_max = max_risk + x_margin
        y_min = min_return - y_margin
        y_max = max_return + y_margin
        
        plt.hlines(solution[0], x_min, x_max, color="orange", linestyle=":", label="success zone")
        plt.vlines(solution[1], y_min, y_max, color="orange", linestyle=":")
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.legend(loc="upper right")
        plt.show()
        
    return evaluations, returns, risks



def asset_robustness(data, sample_size=40, runs=20, generations=50, 
                     test_split=0.3, validation_split=0.3, **kw):
    """
    Evaluate the robustness of the evolutionary algorithm (EA) across 
    multiple random subsets of assets.

    This function simulates different "problem instances" by running the EA 
    on random samples of assets from the full dataset. For each run, it:
      - Samples a random subset of assets
      - Splits the data into train/test sets
      - Lets the EA handle internal train/validation split
      - Trains the EA
      - Evaluates final return on the test set
      - Computes the risk (volatility) from the training data

    Args:
        data (pd.DataFrame): Full historical asset price data.
        sample_size (int, optional): Number of assets to randomly sample 
            in each run. Default is 40.
        runs (int, optional): Number of runs to perform. Default is 20.
        generations (int, optional): EA generations per run. Default is 50.
        test_split (float, optional): Fraction of data to hold out for 
            test evaluation. Default is 0.3.
        validation_split (float, optional): Passed to EA for internal 
            train/validation split. Default is 0.3.
        **kw: Additional keyword arguments passed to the EA constructor.

    Returns:
        tuple:
            - np.ndarray: Test returns (percent) from each run.
            - np.ndarray: Portfolio risk (volatility) from training data.
    """
    returns = np.empty(runs)
    risks = np.empty(runs)

    for i in range(runs):
        
        # Sample a random subset of assets
        data_i = data.sample(sample_size, axis=1)

        # Split into train and test
        split = int((1 - test_split) * len(data_i))
        train_data = data_i[:split]
        test_data = data_i[split:]

        # Train EA (let it do its own train/validation split)
        ea = EA(train_data, validation_split=validation_split, **kw)
        ea.run(generations)

        # Evaluate performance
        test_return, _ = ea.test_returns(test_data, ea.best_chrom)
        train_risk = ea.volatility(ea.best_chrom)

        returns[i] = 100 * test_return
        risks[i] = 100 * train_risk  # scale to percent

        ea.plot_max_fitness()

    return returns, risks



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



def time_robustness(data, periods=5, generations=50, 
                    test_split=0.3, **kw):
    """
    Evaluates the robustness of the evolutionary algorithm across different
    time periods, using a fixed set of assets. For each time chunk, it also
    evaluates a baseline Equal Weight (EW) portfolio for comparison.

    Args:
    - data (pd.DataFrame): Asset closing prices over time.
    - periods (int): Number of time segments. Default is 5.
    - generations (int): Number of generations per EA run. Default is 50.
    - test_split (float): Fraction of each period used as test data. Default is 0.3.
    - **kw: Additional arguments passed to the EA.

    Returns:
    - dict: Dictionary with test returns and risks for both EA and EW across all periods.
    """
    ea_returns = []
    ea_risks = []
    ew_returns = []
    ew_risks = []

    chunk_size = len(data) // periods
    chunks = [data.iloc[i*chunk_size : (i+1)*chunk_size] for i in range(periods - 1)]
    chunks.append(data.iloc[(periods - 1)*chunk_size :])  # final chunk includes remainder

    for i, chunk in enumerate(chunks):
        print(f'\nPeriod {i + 1}/{periods}')

        # Train/test split (70/30 by default)
        split = int((1 - test_split) * len(chunk))
        train_data = chunk[:split]
        test_data = chunk[split:]

        # === Evolutionary Algorithm ===
        ea = EA(train_data, **kw)
        ea.run(generations)
        ret = 100 * ea.test_returns(test_data, ea.best_chrom)[0]
        risk = 100 * ea.volatility(ea.best_chrom)
        ea_returns.append(ret)
        ea_risks.append(risk)

        # === Equal Weight Portfolio ===
        n = train_data.shape[1]
        ew_weights = np.ones(n) / n
        ew_ret = 100 * ea.test_returns(test_data, ew_weights)[0]
        cov_matrix = train_data.pct_change().dropna().cov().values
        ew_risk = 100 * np.sqrt(252 * ew_weights @ cov_matrix @ ew_weights.T)
        ew_returns.append(ew_ret)
        ew_risks.append(ew_risk)

        # Optional: plot EA fitness
        ea.plot_max_fitness(label=f'{chunk.index[0].date()}–{chunk.index[-1].date()}')

    plt.legend(bbox_to_anchor=(1.0, 1.02))
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame({
            "EA_returns": np.array(ea_returns),
            "EA_risks": np.array(ea_risks),
            "EW_returns": np.array(ew_returns),
            "EW_risks": np.array(ew_risks)
        })
    df["Period"] = [f"{chunk.index[0].date()}–{chunk.index[-1].date()}" for chunk in chunks]
    return df.set_index("Period")


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
    split = int(0.5 * len(dta.all_data))
    data = dta.all_data[:split]
    national = data[ticks]
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
