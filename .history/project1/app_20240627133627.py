from flask import Flask, render_template, jsonify, request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

app = Flask(__name__)

# yfinance pandas data reader üzerinde kullanmak için gerekli override
yf.pdr_override()

# Tarih aralığını belirleme
end = datetime.now()
start = datetime(end.year - 7, end.month, end.day)

# Veri hazırlama fonksiyonu
def prepare_data(ticker):
    stock = pdr.get_data_yahoo(ticker, start=start, end=end)
    stock.reset_index(inplace=True)
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock.set_index('Date', inplace=True)
    stock.ffill(inplace=True)
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change() * 100
    stock.dropna(inplace=True)
    mean = stock['Daily_Return'].mean()
    std_dev = stock['Daily_Return'].std()
    stock = stock[(stock['Daily_Return'] >= mean - 3*std_dev) & (stock['Daily_Return'] <= mean + 3*std_dev)]
    return stock

# Grafik oluşturma fonksiyonu
def generate_plot(df, plot_type):
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'adj_close':
        plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
        plt.title('Adjusted Close Price')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
    elif plot_type == 'volume':
        plt.plot(df.index, df['Volume'], label='Volume')
        plt.title('Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
    elif plot_type == 'moving_avg':
        plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
        plt.plot(df.index, df['10_day_MA'], label='10-day MA')
        plt.plot(df.index, df['20_day_MA'], label='20-day MA')
        plt.plot(df.index, df['50_day_MA'], label='50-day MA')
        plt.title('Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_plot')
def get_plot():
    ticker = request.args.get('ticker')
    plot_type = request.args.get('plot_type')
    df = prepare_data(ticker)
    plot_url = generate_plot(df, plot_type)
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)