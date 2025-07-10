import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_historical_data(tickers, start, end):
    """
    Downloads and processes historical data from Yahoo Finance.
    Args:
    tickers (list): List of asset symbols.
    start (str): Start date in 'YYYY-MM-DD' format.
    end (str): End date in 'YYYY-MM-DD' format.
    Returns:
    pd.DataFrame: DataFrame with adjusted close prices per asset.
    """
    # Download data from Yahoo Finance
    data = yf.download(tickers, start=start, end=end)['Close']
    # Handle cases with a single asset
    
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        data.columns = [tickers[0]]
    
    # Remove assets with more than 10 NaN values
    data = data.dropna(axis=1, thresh=data.shape[0]-10)
    
    # Fill NaN values with previous or next value
    data = data.ffill().bfill()
    
    return data

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
    - train_anual_returns (pandas.Series): Annualized returns per asset from the training set.
    - test_daily_returns (pandas.DataFrame): Daily returns per asset from the test set.
    - covariance_matrix (pandas.DataFrame): Covariance matrix of training daily returns.
    """
    train = data.iloc[:int(len(data)*0.7)] # to evolve and select the portfolio
    test = data.iloc[int(len(data)*0.7):] # to evaluate how the selected portfolio performs on unseen data

    train_daily_returns = train.pct_change().dropna()
    train_anual_returns = train_daily_returns.mean() * 252

    test_daily_returns = test.pct_change().dropna()
    
    covariance_matrix = train_daily_returns.cov()
    
    return train_anual_returns, test_daily_returns, covariance_matrix



def evaluate_out_of_sample(weights, test_returns, plot=True):
    """
    Evaluates the performance of a fixed portfolio on out-of-sample data.

    Args:
    - weights (np.ndarray): Portfolio weights (1D array).
    - test_prices (pd.DataFrame): Price data for the test period (columns = assets).
    - plot (bool): Whether to plot cumulative return.

    Returns:
    - dict: Dictionary with expected return, actual return, risk, and returns series.
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

    return {
        'expected_return': f'{round(expected_return, 2)}%',
        'actual_return': f'{round(actual_return, 2)}%',
        'risk': f'{round(risk, 2)}%',
    }



all_data = pd.read_csv(f'{CURRENT_DIR}/../data/closing_prices.csv', index_col=0, parse_dates=True)

####### National capitals#######
# Extract tickers from CSV file with national assets
tickers_NC = pd.read_csv(f'{CURRENT_DIR}/../data/CapitalesNacionales.csv')['Tickers'].tolist()
tickers_NC = [tick for tick in tickers_NC if tick in all_data.columns]

######### FIBRAS ##########
# Extract tickers from CSV file with FIBRAS
tickers_F = pd.read_csv(f'{CURRENT_DIR}/../data/FIBRAS.csv')['Tickers'].tolist()
tickers_F = [tick for tick in tickers_F if tick in all_data.columns]

#### International capitals ########
# Extract tickers from CSV file with international assets
tickers_IC = pd.read_csv(f'{CURRENT_DIR}/../data/CapitalesExtranjeros.csv')['Tickers'].tolist()
tickers_IC = [tick for tick in tickers_IC if tick in all_data.columns]

########### Bonddia ############
tickers_BONDDIA = ['BONDDIAPF2.MX']

###### ENERFIN ########
tickers_ENERFIN = ['ENERFINPF2.MX']

tickers_CETES = ['CETES365']

#### All tickers #####
tickers = tickers_NC + tickers_F + tickers_IC + tickers_BONDDIA + tickers_ENERFIN


daily_returns = all_data.pct_change().dropna()
anual_returns = daily_returns.sum() / len(daily_returns) * 252
risks = daily_returns.std()
covariance_matrix = daily_returns.cov()
correlation_matrix = daily_returns.corr()