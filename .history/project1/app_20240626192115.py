!pip install Flask
from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)

# Function to prepare data for a given ticker
def prepare_data(ticker):
    end = datetime.now()
    start = datetime(end.year - 7, end.month, end.day)
    stock = pdr.get_data_yahoo(ticker, start=start, end=end)
    
    # Data preparation steps
    stock.reset_index(inplace=True)
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock.set_index('Date', inplace=True)
    stock.ffill(inplace=True)

    # Calculate moving averages and daily returns
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change() * 100

    # Clean data by removing outliers
    mean = stock['Daily_Return'].mean()
    std_dev = stock['Daily_Return'].std()
    stock = stock[(stock['Daily_Return'] >= mean - 3*std_dev) & (stock['Daily_Return'] <= mean + 3*std_dev)]
    
    return stock

# Flask route for the home page
@app.route('/')
def index():
    tickers = ['JPM', 'BAC', 'WFC', 'C']
    company_names = ['JPMORGAN', 'BANK OF AMERICA', 'Wells Fargo', 'Citigroup']
    dataframes = [prepare_data(ticker) for ticker in tickers]

    # Render the index.html template with dataframes and company names
    return render_template('exp4.html', dataframes=dataframes, company_names=company_names)

if __name__ == '__main__':
    app.run(debug=True)