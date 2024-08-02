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

@app.route('/')
def index():
    # Your entire Python code goes here, including data preparation, model training, and plotting
    # Example code:
    end = datetime.now()
    start = datetime(end.year - 7, end.month, end.day)

    tickers = ['JPM', 'BAC', 'WFC', 'C']
    company_names = ['JPMORGAN', 'BANK OF AMERICA', 'Wells Fargo', 'Citigroup']
    dataframes = [prepare_data(ticker) for ticker in tickers]

    # Generate plots as needed
    plots = []
    for df, company_name in zip(dataframes, company_names):
        # Example of plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
        plt.title(f"{company_name} - Adjusted Close Price")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plot_img = '/static/' + company_name.replace(' ', '_') + '_plot.png'
        plt.savefig('static/' + company_name.replace(' ', '_') + '_plot.png')
        plots.append(plot_img)

    return render_template('index.html', plots=plots)

if __name__ == '__main__':
    app.run(debug=True)