import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker, period="1y", interval="1d"):
    # Download data
    data = yf.download(tickers=ticker, period=period, interval=interval)
    
    # Add features (SMA, RSI, MACD, Bollinger Bands)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # RSI Calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands calculation
    rolling_std = data['Close'].rolling(window=50).std()
    data['BB_upper'] = data['SMA_50'] + (2 * rolling_std)
    data['BB_lower'] = data['SMA_50'] - (2 * rolling_std)


    # Drop rows with missing values
    data.dropna(inplace=True)
    
    return data

def create_features(data):
    # Define the target (1 for price up, 0 for price down)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_upper', 'BB_lower']
    X = data[features]
    y = data['Target']
    return X, y
