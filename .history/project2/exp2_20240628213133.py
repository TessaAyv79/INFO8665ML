# Tessa Ayvazoglu
# 13/06/2024
# ML programing project
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
#
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from datetime import timedelta
from datetime import datetime
import datetime
import plotly.express as px
import matplotlib.pyplot as plt 
from matplotlib.dates import DateFormatter# Add this import statement
import seaborn as sns  # Add this line for Seaborn
from pandas_datareader import data as pdr
import json  # JSON modülünü import ediyoruz
# yfinance kütüphanesini pandas data reader üzerinde kullanma
yf.pdr_override()
#
# Grafik stil ayarları
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
 
 
# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Function to call local CSS sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Provide the path to the style.css file
style_css_path = r"C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\docs\assets\style.css"
local_css(style_css_path)
#
st.sidebar.image(r'C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\image\images2.jpg', use_column_width=True)
#
# JPMorgan Chase & Co. (JPM) - Ticker Symbol: JPM
# Bank of America Corporation (BAC) - Ticker Symbol: BAC
# Wells Fargo & Company (WFC) - Ticker Symbol: WFC
# Citigroup Inc. (C) - Ticker Symbol: C
# Goldman Sachs Group, Inc. (GS) - Ticker Symbol: GS
# Morgan Stanley (MS) - Ticker Symbol: MS
# U.S. Bancorp (USB) - Ticker Symbol: USB
# PNC Financial Services Group, Inc. (PNC) - Ticker Symbol: PNC
# Truist Financial Corporation (TFC) - Ticker Symbol: TFC
# Capital One Financial Corporation (COF) - Ticker Symbol: COF
# Predefined list of popular stock tickers
#
popular_tickers = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF']
#
# Stock tickers combo box
st.sidebar.subheader("STOCK SEEKER WEB APP")
selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)

# Initialize default values for updated start and end dates
updated_start_date = datetime.datetime(2020, 1, 1)
updated_end_date = datetime.datetime.now()
# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime.datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.datetime.now())

# If the user changes the date range, capture the updated values
if start_date != datetime.datetime(2020, 1, 1) or end_date != datetime.datetime.now():
    updated_start_date = start_date
    updated_end_date = end_date

# Analysis type selection
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])

# Display additional information based on user selection
st.sidebar.subheader("Display Additional Information")
selected_options = {
    "Stock Actions": st.sidebar.checkbox("Stock Actions"),
    "Quarterly Financials": st.sidebar.checkbox("Quarterly Financials"),
    "Institutional Shareholders": st.sidebar.checkbox("Institutional Shareholders"),
    "Quarterly Balance Sheet": st.sidebar.checkbox("Quarterly Balance Sheet"),
    "Quarterly Cashflow": st.sidebar.checkbox("Quarterly Cashflow"),
    "Analysts Recommendation": st.sidebar.checkbox("Analysts Recommendation"),
    "Predicted Prices": st.sidebar.checkbox("Predicted Prices")  # Add Predicted Prices option
}


# Now selected_options should be a dictionary
# Continue with the rest of your code...
# Submit button
button_clicked = st.sidebar.button("Analyze")

# Summary button
summary_clicked = st.sidebar.button("Adv.Anlyz")

# Function to handle analysis
# Function to handle analysis
def handle_analysis(selected_stocks, analysis_type, start_date, end_date):
    st.write(f"Received analysis_type: {analysis_type}")  # Debugging için

    # analysis_type doğru formatta mı kontrol edin
    if isinstance(analysis_type, str):
        if analysis_type.startswith('{') and analysis_type.endswith('}'):
            try:
                analysis_type = json.loads(analysis_type)
                st.write("Successfully decoded JSON string for analysis_type")
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON string for analysis_type: {e}")
                return
        else:
            # analysis_type JSON formatında değilse hata göster
            st.write("analysis_type is not a JSON string, using as a direct string for display")
            selected_options = analysis_type
    else:
        selected_options = analysis_type

    if selected_options != "Predicted Prices":
        for selected_stock in selected_stocks:
            display_stock_analysis(selected_stock, selected_options, start_date, end_date)
            display_additional_information(selected_stock, selected_options, start_date, end_date)
    else:
        for selected_stock in selected_stocks:
            display_predicted_prices(selected_stock, start_date, end_date)
         
# Clean Data process
# Prepare and clean data
def prepare_data(selected_stock, start_date, end_date):
    stock = pdr.get_data_yahoo(selected_stock, start=start_date, end=end_date)
    
    # Tarih formatına çevirme ve tarih sütununu indeks olarak ayarlama
    stock.reset_index(inplace=True)
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock.set_index('Date', inplace=True)

    # Eksik değerleri forward fill yöntemiyle doldurma
    stock.ffill(inplace=True)

    # Hareketli ortalamaları hesaplama
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()

    # Günlük getirileri hesaplama
    stock['Daily_Return'] = stock['Adj Close'].pct_change() * 100  # Yüzde olarak günlük getiri

    # Veriyi temizleme: Eksik değerleri kaldırma
    stock.dropna(inplace=True)

    # Aykırı değerleri çıkarma (standart sapmadan daha büyük sapmalara sahip olanları çıkarma)
    mean = stock['Daily_Return'].mean()
    std_dev = stock['Daily_Return'].std()
    stock = stock[(stock['Daily_Return'] >= mean - 3*std_dev) & (stock['Daily_Return'] <= mean + 3*std_dev)]
    return stock
    
# Function to display stock analysis
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = prepare_data(selected_stock, start_date, end_date)
    st.subheader(f"{selected_stock} - {analysis_type}")

    if analysis_type == "Closing Prices":
        fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'{selected_stock} Closing Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Volume":
        fig = px.line(stock_data, x=stock_data.index, y='Volume', title=f'{selected_stock} Volume')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Volume')
        st.plotly_chart(fig)
        
    elif analysis_type == "Moving Averages":
        stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA20'], mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='50-Day MA'))
        fig.update_layout(title=f'{selected_stock} Moving Averages',
                          xaxis_title='Date',
                          yaxis_title='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Daily Returns":
        stock_data['Daily Return'] = stock_data['Close'].pct_change()
        fig = px.line(stock_data, x=stock_data.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Daily Return')
        st.plotly_chart(fig)
        
    elif analysis_type == "Correlation Heatmap":
        # Example: Calculating correlation heatmap for multiple stocks
        selected_stocks = [selected_stock]  # Example: for single stock
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        st.plotly_chart(fig)
        
    elif analysis_type == "Distribution of Daily Changes":
        stock_data['Daily Change'] = stock_data['Close'].diff()
        fig = px.histogram(stock_data['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
        st.plotly_chart(fig)
selected_options = {}  
# Function to display additional information
def display_additional_information(selected_stock, selected_options, start_date=None, end_date=None):
    import json
    
    # Debugging: Print the type and value of selected_options
    st.write(f"Type of selected_options: {type(selected_options)}")
    st.write(f"Value of selected_options: {selected_options}")

    # Eğer selected_options JSON string ise ayrıştır
    if isinstance(selected_options, str):
        try:
            selected_options = json.loads(selected_options)
            st.write("Successfully decoded JSON string for selected_options")
        except json.JSONDecodeError:
            st.error("Error decoding JSON string for selected_options")
            return

    # selected_options artık dict olmalı
    if not isinstance(selected_options, dict):
        st.error("selected_options must be a dictionary after decoding")
        return

    # Stock bilgilerini hazırlama
    #stock = yf.Ticker(selected_stock)
    stock = stock_data[selected_stock]
    # Stock Actions bilgilerini gösterme
    st.subheader(f"{selected_stock} - Stock Actions")
    stock_actions = stock.actions  # Doğru bir şekilde stock actions al
    if not stock_actions.empty:
        st.write(stock_actions)
    else:
        st.write("No data available for Stock Actions")

    # selected_options üzerinden döngü yap ve ilgili bilgiyi göster
    for option, checked in selected_options.items():
        if checked:
            st.subheader(f"{selected_stock} - {option}")
            if option == "Stock Actions":
                display_action = stock.actions
                if not display_action.empty:
                    st.write(display_action)
                else:
                    st.write("No data available for Stock Actions")
            elif option == "Quarterly Financials":
                display_financials = stock.quarterly_financials
                if not display_financials.empty:
                    st.write(display_financials)
                else:
                    st.write("No data available for Quarterly Financials")
            elif option == "Institutional Shareholders":
                display_shareholders = stock.institutional_holders
                if not display_shareholders.empty:
                    st.write(display_shareholders)
                else:
                    st.write("No data available for Institutional Shareholders")
            elif option == "Quarterly Balance Sheet":
                display_balancesheet = stock.quarterly_balance_sheet
                if not display_balancesheet.empty:
                    st.write(display_balancesheet)
                else:
                    st.write("No data available for Quarterly Balance Sheet")
            elif option == "Quarterly Cashflow":
                display_cashflow = stock.quarterly_cashflow
                if not display_cashflow.empty:
                    st.write(display_cashflow)
                else:
                    st.write("No data available for Quarterly Cashflow")
            elif option == "Analysts Recommendation":
                display_analyst_rec = stock.recommendations
                if not display_analyst_rec.empty:
                    st.write(display_analyst_rec)
                else:
                    st.write("No data available for Analysts Recommendation")
            elif option == "Predicted Prices":
                display_predicted_prices(selected_stock, start_date, end_date)
            else:
                st.write(f"Option {option} is not recognized or not implemented.")
# Function to display predicted prices
# Function to display predicted prices using LSTM
# Function to display predicted prices
def display_predicted_prices(selected_stock, start_date, end_date, prediction_days=30):
    st.subheader(f"{selected_stock} - Predicted Prices")
    
    # Download historical data
    #df = yf.download(selected_stock, start=start_date, end=end_date)
    df = prepare_data(selected_stock, start_date, end_date)
  
    # Prepare the data
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate the prediction dates
    prediction_dates = pd.date_range(end=end_date, periods=len(predictions) + 1, freq='B')[1:]
    
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
    fig.update_layout(title=f'{selected_stock} Predicted Prices',
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig)
# Function to detect pivot points
def isPivot(candle, window, df):
    """
    Function that detects if a candle is a pivot/fractal point
    Args:
        candle: Candle index (datetime object)
        window: Number of days before and after the candle to test if pivot
        df: DataFrame containing the stock data
    Returns:
        1 if pivot high, 2 if pivot low, 3 if both, and 0 default
    """
    # Assuming candle is a datetime object
    candle_timestamp = pd.Timestamp(candle)
    if candle_timestamp - datetime.timedelta(days=window) < df.index[0] or candle_timestamp + datetime.timedelta(days=window) >= df.index[-1]:
        return 0

    pivotHigh = 1
    pivotLow = 2
    start_index = candle_timestamp - datetime.timedelta(days=window)
    end_index = candle_timestamp + datetime.timedelta(days=window)
    for i in range((end_index - start_index).days + 1):
        current_date = start_index + datetime.timedelta(days=i)
    
        if 'low' in df.columns and df.loc[candle_timestamp, 'low'] > df.loc[current_date, 'low']:
            pivotLow = 0
        if 'high' in df.columns and df.loc[candle_timestamp, 'high'] < df.loc[current_date, 'high']:
            pivotHigh = 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

# Function to calculate Chaikin Oscillator
def calculate_chaikin_oscillator(data):
    """
    Calculate Chaikin Oscillator using pandas_ta.
    """
    data['ADL'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
    data['Chaikin_Oscillator'] = ta.ema(data['ADL'], length=3) - ta.ema(data['ADL'], length=10)
    return data

# Define the calculate_stochastic_oscillator function
def calculate_stochastic_oscillator(df, period=14):
    """
    Calculate Stochastic Oscillator (%K and %D).
    """
    df['L14'] = df['Low'].rolling(window=period).min()
    df['H14'] = df['High'].rolling(window=period).max() 
    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

def chart_stochastic_oscillator_and_price(ticker, df):
    """
    Plots the stock's closing price with its 50-day and 200-day moving averages,
    and the Stochastic Oscillator (%K and %D) below the price chart.
    """
    plt.figure(figsize=[16, 8])
    plt.style.use('default')
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(16, 8))
    fig.suptitle(ticker, fontsize=16)

    # Plotting the closing price and moving averages on the first subplot
    ax[0].plot(df['Close'], color='black', linewidth=1, label='Close')
    ax[0].plot(df['ma50'], color='blue', linewidth=1, linestyle='--', label='50-day MA')
    ax[0].plot(df['ma200'], color='red', linewidth=1, linestyle='--', label='200-day MA')
    ax[0].set_ylabel('Price [\$]')
    ax[0].grid(True)
    ax[0].legend(loc='upper left')
    ax[0].axes.get_xaxis().set_visible(False)  # Hide X axis labels for the price plot

    # Plotting the Stochastic Oscillator on the second subplot
    ax[1].plot(df.index, df['%K'], color='orange', linewidth=1, label='%K')
    ax[1].plot(df.index, df['%D'], color='grey', linewidth=1, label='%D')
    ax[1].grid(True)
    ax[1].set_ylabel('Stochastic Oscillator')
    ax[1].set_ylim(0, 100)
    ax[1].axhline(y=80, color='b', linestyle='-')  # Overbought threshold
    ax[1].axhline(y=20, color='r', linestyle='-')  # Oversold threshold
    ax[1].legend(loc='upper left')

    # Formatting the date labels on the X-axis
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # Adjust space between the plots

    st.pyplot(fig)  # Display the plot in Streamlit
    return data

def display_technical_summary(selected_stock, start_date, end_date):
    st.subheader(f"{selected_stock} - Technical Summary")
    
    # stock_data = yf.Ticker(selected_stock)
    stock_data = prepare_data(selected_stock, start_date, end_date)
    
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    # Calculate Chaikin Oscillator
    stock_df = calculate_chaikin_oscillator(stock_df)
    stock_df = calculate_stochastic_oscillator(stock_df)

    # Detect pivot points
    window = 5
    stock_df['isPivot'] = stock_df.apply(lambda x: isPivot(x.name, window, stock_df), axis=1)
    stock_df['pointpos'] = stock_df.apply(lambda row: row['Low'] - 1e-3 if row['isPivot'] == 2 else (row['High'] + 1e-3 if row['isPivot'] == 1 else np.nan), axis=1)

    # Plot candlestick with pivots
    fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                                         open=stock_df['Open'],
                                         high=stock_df['High'],
                                         low=stock_df['Low'],
                                         close=stock_df['Close'],
                                         name='Candlestick')])
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['pointpos'], mode='markers',
                             marker=dict(size=5, color="MediumPurple"),
                             name="Pivot"))
    
    fig.update_layout(title=f'{selected_stock} Candlestick Chart with Pivots',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Plot Chaikin Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Chaikin_Oscillator'], mode='lines', name='Chaikin Oscillator'))
    fig.update_layout(title=f'{selected_stock} Chaikin Oscillator',
                      xaxis_title='Date',
                      yaxis_title='Chaikin Oscillator Value')
    st.plotly_chart(fig)
    # Plot Stochastic Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['%K'], mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['%D'], mode='lines', name='%D'))
    fig.update_layout(title=f'{selected_stock} Stochastic Oscillator',
                      xaxis_title='Date',
                      yaxis_title='Stochastic Oscillator Value')
    st.plotly_chart(fig)
# Define the display_advanced_analysis function
def display_advanced_analysis(selected_stock, start_date, end_date):
    st.subheader(f"Advanced Analysis for {selected_stock}")

    # Download historical data
    #df = yf.download(selected_stock, start=start_date, end=end_date)
    df = prepare_data(selected_stock, start_date, end_date)

    # Add Moving Average Convergence Divergence (MACD)
    df['12 Day EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26 Day EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12 Day EMA'] - df['26 Day EMA']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # MACD Buy/Sell Signals
    df['MACD_Buy_Signal'] = np.where(df['MACD'] > df['Signal Line'], df['MACD'], np.nan)
    df['MACD_Sell_Signal'] = np.where(df['MACD'] < df['Signal Line'], df['MACD'], np.nan)

    fig, ax = plt.subplots()
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['Signal Line'], label='Signal Line', color='red')
    ax.scatter(df.index, df['MACD_Buy_Signal'], marker='^', color='g', label='MACD Buy Signal')
    ax.scatter(df.index, df['MACD_Sell_Signal'], marker='v', color='r', label='MACD Sell Signal')
    ax.set_title(f'MACD for {selected_stock}')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # Format       
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    st.pyplot(fig)

    # Add Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # RSI Buy/Sell Signals
    df['RSI_Buy_Signal'] = np.where(df['RSI'] < 30, df['RSI'], np.nan)
    df['RSI_Sell_Signal'] = np.where(df['RSI'] > 70, df['RSI'], np.nan)

    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax.axhline(70, linestyle='--', alpha=0.5, color='green')
    ax.scatter(df.index, df['RSI_Buy_Signal'], marker='^', color='g', label='RSI Buy Signal')
    ax.scatter(df.index, df['RSI_Sell_Signal'], marker='v', color='r', label='RSI Sell Signal')
    ax.set_title(f'RSI for {selected_stock}')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # Format the date
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    st.pyplot(fig)
def stochastic_calculator(selected_stock, start_date, end_date):
    # Download historical data
    #df = yf.download(selected_stock, start=start_date, end=end_date)
    df = prepare_data(selected_stock, start_date, end_date)
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Stochastic Oscillator (%K and %D)
    high14 = df['High'].rolling(window=14).max()
    low14 = df['Low'].rolling(window=14).min()
    df['%K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['%D'] = df['%K'].rolling(window=3).mean()

    fig, axs = plt.subplots(2, figsize=(12, 8), sharex=True)

    # Plotting the closing prices and moving averages
    axs[0].plot(df.index, df['Close'], label='Closing Price', color='blue')
    axs[0].plot(df.index, df['MA50'], label='50-day MA', color='red')
    axs[0].plot(df.index, df['MA200'], label='200-day MA', color='green')
    axs[0].set_ylabel('Price')
    axs[0].legend(loc='upper left')
    axs[0].set_title(f'{selected_stock} - Closing Prices and Moving Averages')

    # Plotting the Stochastic Oscillator
    axs[1].plot(df.index, df['%K'], label='%K', color='blue')
    axs[1].plot(df.index, df['%D'], label='%D', color='red')
    axs[1].axhline(y=20, color='gray', linestyle='--')
    axs[1].axhline(y=80, color='gray', linestyle='--')
    axs[1].set_ylabel('Oscillator')
    axs[1].set_title('Stochastic Oscillator')
    axs[1].legend(loc='upper left')

    date_format = DateFormatter("%Y-%m-%d")
    axs[1].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    
if button_clicked:
    handle_analysis(selected_stocks, analysis_type, updated_start_date, updated_end_date)
elif summary_clicked:
    for selected_stock in selected_stocks:
        display_advanced_analysis(selected_stock, updated_start_date, updated_end_date)
        display_technical_summary(selected_stock, updated_start_date, updated_end_date)   
        handle_analysis(selected_stock, analysis_type, start_date, end_date)
        prepare_data(selected_stock, start_date, end_date)
            display_technical_summary(selected_stock, start_date, end_date)
            display_advanced_analysis(selected_stock, start_date, end_date)   
            stochastic_calculator(selected_stock, start_date, end_date) 

# Execute analysis when button is clicked
if button_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
            handle_analysis(selected_stock, analysis_type, start_date, end_date)
            prepare_data(selected_stock, start_date, end_date)
    else:
        st.sidebar.warning("Please select at least one stock ticker.")

# Execute technical summary when summary button is clicked
if summary_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
            handle_analysis(selected_stock, analysis_type, start_date, end_date)
            prepare_data(selected_stock, start_date, end_date)
            display_technical_summary(selected_stock, start_date, end_date)
            display_advanced_analysis(selected_stock, start_date, end_date)   
            stochastic_calculator(selected_stock, start_date, end_date)
# Define the stochastic_calculator function
    else:
        st.sidebar.warning("Please select at least one stock ticker.")

# Main logic for handling button clicks
# if button_clicked:
#     for selected_stock in selected_stocks:
#         handle_analysis(selected_stock, analysis_type, start_date, end_date)

# if summary_clicked:
#     for selected_stock in selected_stocks:
#         st.subheader(f"Oscillatron Summary for {selected_stock}")
#         stochastic_calculator(selected_stock, start_date, end_date)
# Define the stochastic_calculator function
# Main function
def main():
    st.title("Stock Market Analysis Web App")
    st.sidebar.title("Stock Market Analysis Web App")
    
    # Sidebar description
    st.sidebar.markdown("""
    ## Stock Market Analysis Web App
    
    This web app provides a detailed analysis of stock market data, including closing prices, volume, moving averages, daily returns, correlation heatmap, and more. You can also explore additional information such as stock actions, quarterly financials, institutional shareholders, and analysts' recommendations.
    """)

    # Sidebar image
    #st.sidebar.image(r'C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\image\stock_market.jpg', use_column_width=True)

    # # Date range selection
    # start_date = st.sidebar.date_input("Start Date", updated_start_date)
    # end_date = st.sidebar.date_input("End Date", updated_end_date)

    # Stock ticker selection
    # selected_stock = st.sidebar.selectbox("Select Stock Ticker", popular_tickers)

    # # Analysis type selection
    # analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])

    # # Display additional information checkbox
    # display_additional_info = st.sidebar.checkbox("Display Additional Information")

    # Submit button
    # if st.sidebar.button("Analyze"):
    #     handle_analysis(selected_stock, analysis_type, start_date, end_date)

# Run the main function
if __name__ == "__main__":
    main()