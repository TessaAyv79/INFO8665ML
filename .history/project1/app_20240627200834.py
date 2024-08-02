from flask import Flask, render_template, jsonify, request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

# Jupyter Notebook'ta Flask uygulaması için threading kullanmak gerekebilir.
from threading import Thread

# Flask uygulaması
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
def generate_plot(df, plot_types):
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))

    for plot_type in plot_types:
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
    return '''
    <h1>Stock Data Visualizer</h1>
    <select id="ticker">
        <option value="JPM">JPMORGAN</option>
        <option value="BAC">BANK OF AMERICA</option>
        <option value="WFC">Wells Fargo</option>
        <option value="C">Citigroup</option>
    </select>
    <button onclick="loadGraph(['adj_close'])">Adjusted Close Price</button>
    <button onclick="loadGraph(['volume'])">Volume</button>
    <button onclick="loadGraph(['moving_avg'])">Moving Averages</button>
    <button onclick="loadGraph(['adj_close', 'volume', 'moving_avg'])">All Graphs</button>

    <div id="plot-container">
        <img id="plot" src="" alt="Graph will be displayed here">
    </div>

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script>
        function loadGraph(plotTypes) {
            var ticker = $('#ticker').val();
            $.get('/get_plot', { ticker: ticker, plot_types: plotTypes.join(',') }, function(data) {
                $('#plot').attr('src', 'data:image/png;base64,' + data.plot_url);
            });
        }
    </script>
    '''

@app.route('/get_plot')
def get_plot():
    ticker = request.args.get('ticker')
    plot_types = request.args.get('plot_types').split(',')
    df = prepare_data(ticker)
    plot_url = generate_plot(df, plot_types)
    return jsonify({'plot_url': plot_url})

# Flask uygulamasını çalıştırma
def run_flask_app():
    app.run(debug=False, use_reloader=False)

# Flask uygulamasını ayrı bir thread'de başlatmak
flask_thread = Thread(target=run_flask_app)
flask_thread.start() 