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
from datetime import datetime, timedelta
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Function to call local CSS sheet
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Provide the path to the style.css file
style_css_path = r"C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\docs\assets\style.css"  # Change this path as needed
local_css(style_css_path)

# Sidebar image
st.sidebar.image(r'C:\Users\Admin\Documents\MLAI\INFO8665ML\project2\image\images2.jpg', use_column_width=True)

# Predefined list of popular stock tickers
popular_tickers = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF']

# Stock tickers combo box
st.sidebar.subheader("STOCK SEEKER WEB APP")
selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)

# Initialize default values for updated start and end dates
updated_start_date = datetime(2020, 1, 1)
updated_end_date = datetime.now()

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# If the user changes the date range, capture the updated values
if start_date != datetime(2020, 1, 1) or end_date != datetime.now():
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

# Submit button
button_clicked = st.sidebar.button("Analyze")

# Summary button
summary_clicked = st.sidebar.button("Advanced Analysis")

# Function to handle analysis
def handle_analysis(selected_stock, analysis_type, start_date, end_date):
    if analysis_type != "Predicted Prices":
        display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
        display_additional_information(selected_stock, selected_options)
    else:
        display_predicted_prices(selected_stock, start_date, end_date)

# Function to display stock analysis
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    st.subheader(f"{selected_stock} - {analysis_type}")

    if analysis_type == "Closing Prices":
        st.line_chart(stock_df['Close'], use_container_width=True)
        
    elif analysis_type == "Volume":
        st.line_chart(stock_df['Volume'], use_container_width=True)
        
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        st.line_chart(stock_df[['Close', 'MA20', 'MA50']], use_container_width=True)
        
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        st.line_chart(stock_df['Daily Return'], use_container_width=True)
        
    elif analysis_type == "Correlation Heatmap":
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        st.write("Correlation Heatmap:")
        st.write(corr)
        
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        st.histogram(stock_df['Daily Change'].dropna(), bins=50)

# Function to display additional information
def display_additional_information(selected_stock, selected_options):
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
    fig.add_trace(go.Scatter(x=prediction_dates, y=predictions[:, 0], mode='lines', name='Predicted Price'))
    fig.update_layout(title=f'{selected_stock} Predicted Closing Price', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Main function
def main():
    st.title("Stock Market Analysis Web App")
    st.sidebar.title("Stock Market Analysis Web App")
    
    # Sidebar description
    st.sidebar.markdown("""
    ## Stock Market Analysis Web App
    
    This web app provides a detailed analysis of stock market data, including closing prices, volume, moving averages, daily returns, correlation heatmap, and more. You can also explore additional information such as stock actions, quarterly financials, institutional shareholders, and analysts' recommendations.
    """

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