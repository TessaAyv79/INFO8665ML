# Tessa Ayvazoglu
# 13/06/2024
# ML programing project
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from datetime import timedelta, datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt 
from matplotlib.dates import DateFormatter# Add this import statement
import seaborn as sns  # Add this line for Seaborn
from fbprophet import Prophet
from sklearn.exceptions import InconsistentVersionWarning
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
# Sidebar image
st.sidebar.image(r'C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\image\images2.jpg', use_column_width=True)
st.header("STOCK SEEKER WEB APP")
# # Centered title using HTML and CSS
# st.markdown(
#     f"""
#     <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: white; font-size: 24px; font-weight: bold;">
#         STOCK SEEKER WEB APP
#     </div>
#     """, unsafe_allow_html=True)
# Image path
# Main image display
image_path = r"C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\image\Screenshot 2024-06-28 183240.png"
st.image(image_path, use_column_width=True)

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

# Stock tickers combo box
st.sidebar.subheader("STOCK SEEKER WEB APP")

selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)
# Stock tickers combo box
# Merging the stock lists and removing duplicates

# Update the slider to allow up to 10 years of prediction
n_years = st.sidebar.slider('Years of prediction:', 1, 10)
prd = f'{n_years}y'
# Örnek olarak, 1 yıl, 2 yıl, 3 yıl gibi geçerli period değerlerini kontrol edin
valid_periods = ['1y', '2y', '3y', '5y', 'max']


def validate_period(prd, valid_periods):
    if prd not in valid_periods:
        st.warning(f"Invalid period selected: {prd}. Please select a valid period from {', '.join(valid_periods)}.")
        return None
    return prd
  
    # Correct placement of return statement inside the function
period = n_years * 365
TODAY = datetime.today()
start_datex = TODAY - timedelta(days=period)
# Initialize default values for updated start and end dates
# updated_start_date = datetime.datetime(2020, 1, 1)
# updated_end_date = datetime.datetime.now()
# Date range selection
start_date = st.sidebar.date_input("Start Date", start_datex)
end_date = st.sidebar.date_input("End Date", datetime.today())
updated_start_date = start_date
updated_end_date = end_date
# If the user changes the date range, capture the updated values
# if start_date != datetime(2020, 1, 1) or end_date != datetime.today():
#     # Perform some action based on the condition
#     st.write("Dates are different from specified defaults or current date.")
# else:
#     st.write("Dates are default: Start Date is January 1, 2020 and End Date is today's date.")

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

# Submit button
button_clicked = st.sidebar.button("Analyze")

# Summary button
summary_clicked = st.sidebar.button("Adv.Anlyz")

# Documentation and Help Section
st.sidebar.title('Help & Documentation')
st.sidebar.write("Welcome to the Stock Forecast App!")
st.sidebar.write("To use the app, select a stock from the dropdown menu, choose the forecast period using the slider, and view the forecast plot and components.")
st.sidebar.write("Please note that the forecasts provided are based on historical data and may not accurately predict future stock prices. Use them for informational purposes only.")

# # Feedback Mechanism
# st.sidebar.title('Feedback & Support')
# feedback_form = st.sidebar.form(key='feedback_form')
# feedback_text = feedback_form.text_area('Share your feedback or report an issue:', height=150)
# email_input = feedback_form.text_input('Your email (optional):')
# submit_button = feedback_form.form_submit_button(label='Submit Feedback')

# if submit_button:
#     if feedback_text.strip() == '':
#         st.sidebar.error("Feedback cannot be empty. Please provide your feedback.")
#     else:
#         # Send feedback to a designated email address or store it in a database
#         st.sidebar.success("Thank you for your feedback! We'll review it and take necessary actions.")
#         if email_input.strip() != '':
#             # Send email notification if email is provided
#             st.sidebar.write(f"A confirmation email has been sent to {email_input}.")

@st.cache_resource
def load_data(selected_stocks, start_date, end_date):
    try:
        if not selected_stocks:
            st.warning("Please select at least one stock ticker.")
            return None
        
        data = yf.download(selected_stocks, start_date, end_date)
        
        if data.empty:
            st.warning("No data found for selected stocks.")
            return None
        
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Failed to fetch data for {selected_stocks}. Error: {str(e)}")
        return None
data_load_state = st.text('Loading data...')
data = load_data(selected_stocks, start_date, end_date)
if data is not None:
    data_load_state.text('Loading data... done!')
    st.subheader('Raw data')
    st.write(data.tail())
# Function to plot raw data
    def plot_raw_data(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.update_layout(title='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

# Function to predict forecast with Prophet
    @st.cache_resource
    def predict_forecast(df_train, period):
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        return m, forecast
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m, forecast = predict_forecast(df_train, period)

   # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    
# def prepare_data(ticker):
#     # Veri yükleme ve hazırlama işlemlerini bu fonksiyonda yapın
#     stock_data = yf.Ticker(ticker)
#     df = stock_data.history(period='1y') 
#     # prd = f'{n_years}y'
#     # df = stock_data.history(period=prd) # Son 1 yılın verilerini alıyoruz
#     df['10_day_MA'] = df['Close'].rolling(window=10).mean()
#     df['20_day_MA'] = df['Close'].rolling(window=20).mean()
#     df['50_day_MA'] = df['Close'].rolling(window=50).mean()
#     df['Daily_Return'] = df['Close'].pct_change()
#     df.dropna(inplace=True)  # NaN değerleri temizliyoruz
#     return df 

def prepare_data(ticker, validated_period):
    try:
        # Load and prepare data
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period=validated_period)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker} with period {validated_period}")
        
        df['10_day_MA'] = df['Close'].rolling(window=10).mean()
        df['20_day_MA'] = df['Close'].rolling(window=20).mean()
        df['50_day_MA'] = df['Close'].rolling(window=50).mean()
        
        # Drop NaN values after calculating moving averages
        df.dropna(inplace=True)  
        
        # Calculate daily returns after ensuring no NaN values
        df['Daily_Return'] = df['Close'].pct_change()
        
        return df
    
    except Exception as e:
        # Handle exceptions, print or log the error
        print(f"Error while preparing data for {ticker}: {str(e)}")
        # Optionally, raise the exception again if needed
        raise e
def train_and_evaluate_model(ticker, company_name, validated_period):
    df = prepare_data(ticker, validated_period)
    X = df[['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']]
    y = df['Close']  # 'Adj Close' yerine 'Close' kullanıyoruz, çünkü 'Adj Close' her zaman mevcut olmayabilir

    # Veri bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Veri ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest model
    rf = RandomForestRegressor()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test_scaled)

    # Gradient Boosting model
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train_scaled, y_train)
    y_pred_gbr = gbr.predict(X_test_scaled)

    # Model değerlendirme
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)

    # Sonuçları yazdırma
    st.write(f"{company_name} ({ticker}):")
    st.write(f"Random Forest - MSE: {mse_rf}, MAE: {mae_rf}")
    st.write(f"Gradient Boosting - MSE: {mse_gbr}, MAE: {mae_gbr}")

    # Tahmin grafikleri
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', alpha=0.7)
    ax.plot(df.index[-len(y_test):], y_pred_rf, label='Predicted Prices (RF)', alpha=0.7)
    ax.plot(df.index[-len(y_test):], y_pred_gbr, label='Predicted Prices (GBR)', alpha=0.7)
    ax.legend()
    ax.set_title(f'Actual vs Predicted Stock Prices for {company_name} ({ticker})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
# Main Streamlit app
# LSTM modeli eğitimi ve değerlendirme fonksiyonu
def train_and_evaluate_lstm_model1(ticker, company_name, validated_period):
    
    df = prepare_data(ticker, validated_period)
    X = df[['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']]
    y = df['Close']  # 'Adj Close' yerine 'Close' kullanıyoruz, çünkü 'Adj Close' her zaman mevcut olmayabilir

    # Veri bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Veri ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LSTM için veri şekillendirme
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # LSTM modeli oluşturma
    lstm_model = Sequential([
        LSTM(100, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(1)
    ])

    # Model derleme
    lstm_model.compile(optimizer='adam', loss='mse')

    # Model eğitimi
    history = lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # Tahmin yapma
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

    # Model performansını değerlendirme
    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

    st.write(f"{company_name} ({ticker}) - LSTM Model:")
    st.write(f"MSE: {mse_lstm}, MAE: {mae_lstm}")

    # Tahmin grafikleri
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', alpha=0.7)
    ax.plot(df.index[-len(y_test):], y_pred_lstm, label='Predicted Prices (LSTM)', alpha=0.7)
    ax.legend()
    ax.set_title(f'Actual vs Predicted Stock Prices for {company_name} ({ticker}) - LSTM')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Function to handle analysis
def handle_analysis(selected_stock, analysis_type, start_date, end_date, validate_period):
    if analysis_type != "Predicted Prices":
        display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
        display_additional_information(selected_stock, selected_options)
    else:
        display_predicted_prices(selected_stock, start_date, end_date)
        for selected_stock in selected_stocks:
            company_name = "Company Name"  # Şirket ismini uygun bir şekilde alın veya belirleyin
            train_and_evaluate_model(selected_stock, company_name, validate_period)
            train_and_evaluate_lstm_model1(selected_stock, company_name, validate_period)
            train_and_evaluate_dnn_model(selected_stock, company_name, validate_period)

# Function to display stock analysis
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    st.subheader(f"{selected_stock} - {analysis_type}")

    if analysis_type == "Closing Prices":
        fig = px.line(stock_df, x=stock_df.index, y='Close', title=f'{selected_stock} Closing Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Volume":
        fig = px.line(stock_df, x=stock_df.index, y='Volume', title=f'{selected_stock} Volume')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Volume')
        st.plotly_chart(fig)
        
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA20'], mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA50'], mode='lines', name='50-Day MA'))
        fig.update_layout(title=f'{selected_stock} Moving Averages',
                          xaxis_title='Date',
                          yaxis_title='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        fig = px.line(stock_df, x=stock_df.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Daily Return')
        st.plotly_chart(fig)
        
    elif analysis_type == "Correlation Heatmap":
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        st.plotly_chart(fig)
        
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig = px.histogram(stock_df['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
        st.plotly_chart(fig)

# Function to display additional information
def display_additional_information(selected_stock, selected_options, validate_period):
    for option, checked in selected_options.items():
        if checked:
            st.subheader(f"{selected_stock} - {option}")
            if option == "Stock Actions":
                display_action = yf.Ticker(selected_stock).actions
                if not display_action.empty:
                    st.write(display_action)
                else:
                    st.write("No data available")
            elif option == "Quarterly Financials":
                display_financials = yf.Ticker(selected_stock).quarterly_financials
                if not display_financials.empty:
                    st.write(display_financials)
                else:
                    st.write("No data available")
            elif option == "Institutional Shareholders":
                display_shareholders = yf.Ticker(selected_stock).institutional_holders
                if not display_shareholders.empty:
                    st.write(display_shareholders)
                else:
                    st.write("No data available")
            elif option == "Quarterly Balance Sheet":
                display_balancesheet = yf.Ticker(selected_stock).quarterly_balance_sheet
                if not display_balancesheet.empty:
                    st.write(display_balancesheet)
                else:
                    st.write("No data available")
            elif option == "Quarterly Cashflow":
                display_cashflow = yf.Ticker(selected_stock).quarterly_cashflow
                if not display_cashflow.empty:
                    st.write(display_cashflow)
                else:
                    st.write("No data available")
            elif option == "Analysts Recommendation":
                display_analyst_rec = yf.Ticker(selected_stock).recommendations
                if not display_analyst_rec.empty:
                    st.write(display_analyst_rec)
                else:
                    st.write("No data available")
            elif option == "Predicted Prices":
                display_predicted_prices(selected_stock, start_date, end_date)
                for selected_stock in selected_stocks:
                    company_name = "Company Name"  # Şirket ismini uygun bir şekilde alın veya belirleyin
                    train_and_evaluate_model(selected_stock, company_name, validate_period)
                    train_and_evaluate_lstm_model1(selected_stock, company_name, validate_period)
                    train_and_evaluate_dnn_model(selected_stock, company_name, validate_period)
# DNN modeli eğitimi ve değerlendirme fonksiyonu
def train_and_evaluate_dnn_model(ticker, company_name, validated_period):
    df = prepare_data(ticker, validated_period)
    X = df[['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']]
    y = df['Close']  # 'Adj Close' yerine 'Close' kullanıyoruz, çünkü 'Adj Close' her zaman mevcut olmayabilir

    # Veri bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Veri ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # DNN modeli oluşturma
    dnn_model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    dnn_model.compile(optimizer='adam', loss='mse')

    # Model eğitimi
    history = dnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    # Tahmin yapma
    y_pred_dnn = dnn_model.predict(X_test_scaled).flatten()

    # Model performansını değerlendirme
    mse_dnn = mean_squared_error(y_test, y_pred_dnn)
    mae_dnn = mean_absolute_error(y_test, y_pred_dnn)

    st.write(f"{company_name} ({ticker}) - DNN Model:")
    st.write(f"MSE: {mse_dnn}, MAE: {mae_dnn}")

    # Tahmin grafikleri
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', alpha=0.7)
    ax.plot(df.index[-len(y_test):], y_pred_dnn, label='Predicted Prices (DNN)', alpha=0.7)
    ax.legend()
    ax.set_title(f'Actual vs Predicted Stock Prices for {company_name} ({ticker}) - DNN')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)                     
# Function to display predicted prices
# Function to display predicted prices using LSTM
# Function to display predicted prices
def display_predicted_prices(selected_stock, start_date, end_date, prediction_days=30):
    st.subheader(f"{selected_stock} - Predicted Prices")
    
    # Download historical data
    df = yf.download(selected_stock, start=start_date, end=end_date)
    
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
    if candle_timestamp - timedelta(days=window) < df.index[0] or candle_timestamp + timedelta(days=window) >= df.index[-1]:
        return False
        return 0

    pivotHigh = 1
    pivotLow = 2
    start_index = candle_timestamp - timedelta(days=window)
    end_index = candle_timestamp + timedelta(days=window)
    for i in range((end_index - start_index).days + 1):
        current_date = start_index + timedelta(days=i)
    
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
    
    stock_data = yf.Ticker(selected_stock)
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
    df = yf.download(selected_stock, start=start_date, end=end_date)

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
    df = yf.download(selected_stock, start=start_date, end=end_date)
    
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

# Execute analysis when button is clicked
if button_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
            handle_analysis(selected_stock, analysis_type, start_date, end_date, validate_period)
    else:
        st.sidebar.warning("Please select at least one stock ticker.")

# Execute technical summary when summary button is clicked
if summary_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
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
# Email
# Function to send email
# Function to handle email sending (ensure to add security checks and encryption)
# Function to send email (ensure to add security checks and encryption)
# Function to send email

 
# Feedback Mechanism
st.sidebar.title('Feedback & Support')
feedback_form = st.sidebar.form(key='feedback_form')
feedback_text = feedback_form.text_area('Share your feedback or report an issue:', height=150)
email_input = feedback_form.text_input('Your email (optional):')
submit_button = feedback_form.form_submit_button(label='Submit Feedback')


if submit_button:
    if feedback_text.strip() == '':
        st.sidebar.error("Feedback cannot be empty. Please provide your feedback.")
    else:
        # Send feedback to a designated email address or store it in a database
        st.sidebar.success("Thank you for your feedback! We'll review it and take necessary actions.")
        if email_input.strip() != '':
            # Send email notification if email is provided
            st.sidebar.write(f"A confirmation email has been sent to {email_input}.")
