import pandas as pd
import yfinance as yf
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
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    # Handle cases with a single asset
    
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        data.columns = [tickers[0]]
    
    # Remove assets with more than 10 NaN values
    # data = data.dropna(axis=1, thresh=data.shape[0]-10)
    
    # Fill NaN values with previous or next value
    # data = data.ffill().bfill()
    
    return data

try:
    all_data = pd.read_csv("../data/closing_prices.csv", index_col=0, parse_dates=True)
except:
    start = '2021-01-01'
    end = '2024-04-05'
    
    tickers = pd.read_csv("../data/sp100_2021.csv")['Ticker'].to_list()
    all_data = get_historical_data(tickers, start, end)
    all_data.dropna(axis=1, thresh=100, inplace=True)
    
    file_path = os.path.join(CURRENT_DIR, "../data/closing_prices.csv")
    all_data.to_csv(file_path, index=True)