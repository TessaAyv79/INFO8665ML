# app.py

from flask import Flask, render_template, request, redirect, url_for
from utils import prepare_data, analyze_data, plot_pairplot_and_heatmap, train_and_evaluate_model, train_and_evaluate_lstm_model, train_and_evaluate_dnn_model
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Tanımlamalar
tickers = ['JPM', 'BAC', 'WFC', 'C']
company_names = ['JPMORGAN', 'BANK OF AMERICA', 'Wells Fargo', 'Citigroup']

# Anasayfa rotası
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Formdan gelen verileri al
        selected_ticker = request.form.get('ticker')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        selected_criteria = request.form.getlist('criteria')

        # Seçilen hisse senedini ve tarih aralığını kullanarak veriyi hazırla
        df = prepare_data(selected_ticker)

        # Veriyi filtrele
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Seçilen kriterlere göre analiz yap
        graphs = []
        if 'adjusted_close' in selected_criteria:
            fig, ax = plt.subplots()
            df['Adj Close'].plot(ax=ax, title=f"{selected_ticker} - Adjusted Close Price")
            ax.set_xlabel('Date')
            ax.set_ylabel('Adjusted Close Price')
            graphs.append(encode_plot(fig))

        if 'volume' in selected_criteria:
            fig, ax = plt.subplots()
            df['Volume'].plot(ax=ax, title=f"{selected_ticker} - Volume")
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            graphs.append(encode_plot(fig))

        if 'moving_averages' in selected_criteria:
            fig, ax = plt.subplots()
            df[['Adj Close', '10_day_MA', '20_day_MA', '50_day_MA']].plot(ax=ax, title=f"{selected_ticker} - Moving Averages")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            graphs.append(encode_plot(fig))

        # Daha fazla analiz eklenebilir

        return render_template('index.html', tickers=tickers, graphs=graphs, company_names=company_names)
    
    return render_template('index.html', tickers=tickers, company_names=company_names)

def encode_plot(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read())
    return string.decode("utf-8")

if __name__ == '__main__':
    app.run(debug=True)