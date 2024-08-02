
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

def prepare_data(ticker):
    # Implementation of prepare_data function

@app.route('/')
def index():
    end = datetime.now()
    start = datetime(end.year - 7, end.month, end.day)

    tickers = ['JPM', 'BAC', 'WFC', 'C']
    company_names = ['JPMORGAN', 'BANK OF AMERICA', 'Wells Fargo', 'Citigroup']
    dataframes = [prepare_data(ticker) for ticker in tickers]

    # Rest of your Flask route handling code
    # ...

if __name__ == '__main__':
    app.run(debug=True)