import yfinance as yf
import pandas as pd
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