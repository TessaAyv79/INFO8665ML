from flask import Flask, render_template
from my_data_module import prepare_data  # Adjust module name as per your actual structure

app = Flask(__name__)

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