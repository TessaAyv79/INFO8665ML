 
import matplotlib.dates as mdates
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import warnings
from streamlit.errors import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

style_css_path = r"C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\docs\assets\style.css"
local_css(style_css_path)  

st.sidebar.image(r'C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\image\images2.jpg', use_column_width=True)
st.header("STOCK SEEKER WEB APP")

image_path = r"C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\image\Screenshot 2024-06-28 183240.png"
st.image(image_path, use_column_width=True)

popular_tickers = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF']
st.sidebar.subheader("STOCK SEEKER WEB APP")
selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)

n_years = st.sidebar.slider('Years of prediction:', 1, 10)
prd = f'{n_years}y'

valid_periods = [f'{i}y' for i in range(1, 101)] + ['max']

def validate_period(prd, valid_periods):
    if prd not in valid_periods:
        st.warning(f"Invalid period selected: {prd}. Please select a valid period from {', '.join(valid_periods)}.")
        return None
    return prd

period = n_years * 365
TODAY = datetime.today()
start_datex = TODAY - timedelta(days=period)
start_date = st.sidebar.date_input("Start Date", start_datex)
end_date = st.sidebar.date_input("End Date", datetime.today())
updated_start_date = start_date
updated_end_date = end_date

analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])

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

button_clicked = st.sidebar.button("Analyze")
summary_clicked = st.sidebar.button("Adv.Anlyz")

st.sidebar.title('Help & Documentation')
st.sidebar.write("Welcome to the Stock Forecast App!")
st.sidebar.write("To use the app, select a stock from the dropdown menu, choose the forecast period using the slider, and view the forecast plot and components.")
st.sidebar.write("Please note that the forecasts provided are based on historical data and may not accurately predict future stock prices. Use them for informational purposes only.")

# Financial Education Resources Section
st.sidebar.title('Financial Education Resources')
st.sidebar.write("Learn more about financial analysis and stock market strategies with these resources:")
st.sidebar.write("- [Investopedia](https://www.investopedia.com/)")
st.sidebar.write("- [Yahoo Finance Education](https://finance.yahoo.com/education/)")
st.sidebar.write("- [Morningstar Classroom](https://www.morningstar.com/education)")

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

    def plot_raw_data(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.update_layout(title='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

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

    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

def get_stock_data(stock_ticker, start_date, end_date):
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    return data

def get_company_name(selected_stock):
    try:
        ticker = yf.Ticker(selected_stock)
        company_name = ticker.info.get('longName', 'Unknown Company')
        return company_name
    except Exception as e:
        st.error(f"Failed to get company name for {selected_stock}. Error: {str(e)}")
        return 'Unknown Company'

company_name = get_company_name(selected_stocks)

def prepare_data(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    df = stock_data.history(start=start_date, end=end_date)
    return df

def train_and_evaluate_model(selected_stock, df, company_name):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test_scaled)

    gbr = GradientBoostingRegressor()
    gbr.fit(X_train_scaled, y_train)
    y_pred_gbr = gbr.predict(X_test_scaled)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)

    st.write(f"{company_name} ({selected_stock}):")
    st.write(f"Random Forest - MSE: {mse_rf}, MAE: {mae_rf}")
    st.write(f"Gradient Boosting - MSE: {mse_gbr}, MAE: {mae_gbr}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', alpha=0.7)
    ax.plot(df.index[-len(y_test):], y_pred_rf, label='Predicted Prices (RF)', alpha=0.7)
    ax.plot(df.index[-len(y_test):], y_pred_gbr, label='Predicted Prices (GBR)', alpha=0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title(f'{company_name} - Actual vs Predicted Prices')
    ax.legend()
    st.pyplot(fig)

if button_clicked:
    for selected_stock in selected_stocks:
        st.subheader(f"Analysis Results for {selected_stock}")
        stock_df = prepare_data(selected_stock, updated_start_date, updated_end_date)
        train_and_evaluate_model(selected_stock, stock_df, company_name)

if summary_clicked:
    st.subheader("Advanced Analysis Summary")
    st.write("To be continued")