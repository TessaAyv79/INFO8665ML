from flask import Flask, jsonify, request
from utils import prepare_data, generate_plot

# Flask uygulaması
app = Flask(__name__)

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