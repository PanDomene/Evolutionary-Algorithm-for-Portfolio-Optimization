import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import data as dta
from .algorithms import EA, equal_weight
from .testing import evaluate_out_of_sample

def MBF(train_data, test_data, runs=10, generations=50, verbose=True, **kwargs):
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
        best = ea.best_chrom
        actual_return, _ = ea.test_returns(test_data, best)
        cov_matrix = train_data.pct_change().dropna().cov().values
        risk = np.sqrt(252 * best @ cov_matrix @ best)

        actual_returns[i] = actual_return
        risks[i] = risk
    if verbose:
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
    cov_matrix = train_data.pct_change().dropna().cov().values

    for i in range(runs):
        ea = EA(train_data, **kwargs)
        gen = 0

        while gen < max_gens:
            ea.run(1)
            
            best = ea.best_chrom
            # Evaluate return on test set
            actual_return  = 100 * ea.test_returns(test_data, best)[0] 
            # Evaluate risk on training set
            estimated_risk = 100 * np.sqrt(252 * best @ cov_matrix @ best)
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
        c1 = "limegreen"  # color for successful runs
        c2 = "gray"  # color for failed runs
        colors = [c1 if (ret > solution[0] and rsk < solution[1]) 
                  else c2 for ret, rsk in zip(returns, risks)]
        
        plt.figure(figsize=(5, 5))
        plt.scatter(risks, returns, c=colors, alpha=0.7, edgecolors='black', s=50)
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
        
        plt.hlines(solution[0], x_min, solution[1], color="red", linestyle=":", label="success zone")
        plt.vlines(solution[1], solution[0], y_max, color="red", linestyle=":")
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")
        plt.show()
        
    return evaluations, returns, risks



def asset_robustness(data, sample_size=50, runs=20, generations=50, 
                     test_split=0.3, validation_split=0.3, **kw):
    """
    Evaluate the robustness of the evolutionary algorithm (EA) across 
    multiple random subsets of assets, comparing the result to the 
    equal-weight portfolio for the given set of assets.

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
            - np.ndarray: Return improvement over the equal-weight portfolio (in %).
            - np.ndarray: Risk improvement over the equal-weight portfolio (in %).
    """
    returns = np.empty(runs)
    risks = np.empty(runs)
    ew_returns = np.empty(runs)
    ew_risks = np.empty(runs)

    # Initialize arrays to store improvements over the equal-weight portfolio
    return_improvement = np.empty(runs)
    risk_improvement = np.empty(runs)

    # Loop over the number of runs
    for i in range(runs):
        
        # Sample a random subset of assets
        data_i = data.sample(sample_size, axis=1)

        # Split into train and test
        split = int((1 - test_split) * len(data_i))
        train_data = data_i[:split]
        test_data = data_i[split:]
        cov_matrix = train_data.pct_change().dropna().cov().values

        # Evaluate the equal-weight portfolio for comparison
        ew_portfolio = equal_weight(train_data)
        ew_ret = 100 * evaluate_out_of_sample(ew_portfolio, test_data, plot=False, verbose=False)[0]
        ew_rsk = 100 * np.sqrt(252 * ew_portfolio @ cov_matrix @ ew_portfolio)

        ew_returns[i] = ew_ret
        ew_risks[i] = ew_rsk

        # Run the EA on the sampled data
        ea = EA(train_data, validation_split=validation_split, max_risk=0.9*ew_rsk/100, **kw)
        ea.run(generations)
        best = ea.best_chrom

        # Evaluate the best individual on the test set
        ret = 100 * ea.test_returns(test_data, best)[0]
        rsk = 100 * np.sqrt(252 * best @ cov_matrix @ best)
        
        # Store the results
        returns[i] = ret
        risks[i] = rsk
    
        # Calculate improvements over the equal-weight portfolio
        return_improvement[i] = 100 * (ret - ew_ret)/ np.abs(ew_ret)
        risk_improvement[i] = 100 * (ew_rsk - rsk) / np.abs(ew_rsk)

    # Check if the EA outperformed the equal-weight portfolio
    success = np.sum((returns > ew_returns) & (risks < ew_risks))

    if success > 0:
        print(f"EA outperformed EW in {success} out of {runs} runs.")
    else:
        print("EA did not outperform the equal-weight portfolio in any run.")

    df = pd.DataFrame({
            "EA_returns": returns,
            "EW_returns": ew_returns,
            "EA_risks": risks,
            "EW_risks": ew_risks,
            "Return_improvement": return_improvement,
            "Risk_improvement": risk_improvement
        })
    
    return df



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
    Evaluate the robustness of the evolutionary algorithm (EA) across 
    different time periods, using a fixed set of assets, and compare it 
    to the equal-weight portfolio (EW) over each period.

    Args:
        data (pd.DataFrame): Asset closing prices over time.
        periods (int): Number of time segments. Default is 5.
        generations (int): Number of generations per EA run. Default is 50.
        test_split (float): Fraction of each period used as test data. Default is 0.3.
        **kw: Additional keyword arguments passed to the EA constructor.

    Returns:
        pd.DataFrame: Table of results with returns, risks, and improvements 
        over EW for each period.
    """
    ea_returns = []
    ea_risks = []
    ew_returns = []
    ew_risks = []
    return_improvement = []
    risk_improvement = []
    periods_labels = []

    # Split the data into roughly equal-length periods
    chunk_size = len(data) // periods
    chunks = [data.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(periods - 1)]
    chunks.append(data.iloc[(periods - 1) * chunk_size:])  # last chunk gets remainder

    for chunk in chunks:
        # Label the period
        periods_labels.append(f"{chunk.index[0].date()}–{chunk.index[-1].date()}")

        # Split into train and test
        split = int((1 - test_split) * len(chunk))
        train_data = chunk[:split]
        test_data = chunk[split:]
        cov_matrix = train_data.pct_change().dropna().cov().values

        # Equal-weight portfolio
        n = train_data.shape[1]
        ew_weights = np.ones(n) / n
        ew_ret = 100 * evaluate_out_of_sample(ew_weights, test_data, plot=False, verbose=False)[0]
        ew_rsk = 100 * np.sqrt(252 * ew_weights @ cov_matrix @ ew_weights)

        ew_returns.append(ew_ret)
        ew_risks.append(ew_rsk)

        # EA run
        ea = EA(train_data, max_risk=1.1*ew_rsk / 100, **kw)
        ea.run(generations)
        best = ea.best_chrom
        ret = 100 * ea.test_returns(test_data, best)[0]
        rsk = 100 * np.sqrt(252 * best @ cov_matrix @ best)
        
        ea_returns.append(ret)
        ea_risks.append(rsk)

        # Improvements
        return_improvement.append((ret - ew_ret) / np.abs(ew_ret) * 100)
        risk_improvement.append((ew_rsk - rsk) / np.abs(ew_rsk) * 100)

    # Compute success count
    returns_arr = np.array(ea_returns)
    risks_arr = np.array(ea_risks)
    ew_returns_arr = np.array(ew_returns)
    ew_risks_arr = np.array(ew_risks)

    success = np.sum((returns_arr > ew_returns_arr) & (risks_arr < ew_risks_arr))

    if success > 0:
        print(f"EA outperformed EW in {success} out of {periods} periods.")
    else:
        print("EA did not outperform the equal-weight portfolio in any period.")

    df = pd.DataFrame({
        "EA_returns": returns_arr,
        "EW_returns": ew_returns_arr,
        "EA_risks": risks_arr,
        "EW_risks": ew_risks_arr,
        "Return_improvement": np.array(return_improvement),
        "Risk_improvement": np.array(risk_improvement)
    }, index=periods_labels)

    df.index.name = "Period"
    return df


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
