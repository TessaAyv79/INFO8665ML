#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


# Install necessary libraries
get_ipython().system('pip install -q yfinance')
get_ipython().system('pip install pandas-datareader')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import streamlit as st


# 

# In[2]:


# Load/Read Data
yf.pdr_override()

# Set plotting styles
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# Define company tickers
tech_list = ['NVDA']

# Download stock data for the past year
end = datetime.now()
start = datetime(end.year - 7, end.month, end.day)
company_list = []
for stock in tech_list:
    company_list.append(yf.download(stock, start=start, end=end))

company_name = ["NVIDIA"]


# 

# In[194]:


# Fill missing values using forward fill
for company in company_list:
    company.ffill(inplace=True)

# Ensure consistent date format
for company in company_list:
    company.reset_index(inplace=True)
    company['Date'] = pd.to_datetime(company['Date'])
    company.set_index('Date', inplace=True)

# Add company name column to each dataframe
for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

# Concatenate individual stock data into a single DataFrame
df = pd.concat(company_list, axis=0)

# Shuffle the data and get a random sample of the last 10 rows
df = df.sample(frac=1).reset_index(drop=True)
print(df.tail(10))

df = df.reset_index()
df = df.fillna(method='ffill')


# 

# In[4]:


# Plotting closing prices
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.title(f"Closing Price of {tech_list[i - 1]}")
plt.tight_layout()

# Plotting sales volume
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.title(f"Sales Volume for {tech_list[i - 1]}")
plt.tight_layout()

ma_day = [10, 20, 50]


# 

# In[13]:


# Calculate moving averages
ma_day = [10, 20, 50]
for company_data in company_list:
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        company_data[column_name] = company_data['Adj Close'].rolling(ma).mean()

# plt.figure(figsize=(15, 10))
# plt.subplots_adjust(top=1.25, bottom=1.2)
for i, company_data in enumerate(company_list, 1):
    # plt.subplot(2, 2, i)
    company_data[['Adj Close', f"MA for {ma_day[0]} days", f"MA for {ma_day[1]} days", f"MA for {ma_day[2]} days"]].plot()
    plt.title(f"Moving Averages for {tech_list[i-1]}")
plt.tight_layout()
plt.show()


# 

# In[17]:


# Calculate daily returns
for company_data in company_list:
    company_data['Daily Return'] = company_data['Adj Close'].pct_change()

# Plotting daily returns
plt.figure(figsize=(15, 10))
company_list[0]['Daily Return'].plot(legend=True, linestyle='--', marker='o')
plt.title(f"Daily Return of {tech_list[0]}")
plt.tight_layout()
plt.show()


# 

# In[18]:


# Günlük getirilerin dağılımını çizme
plt.figure(figsize=(12, 9))
company_data = company_list[0]
company_data['Daily Return'].hist(bins=50)
plt.xlabel('Daily Return')
plt.title(f'Distribution of Daily Return for {tech_list[0]}')
plt.tight_layout()
plt.show()

# Tek bir şirket için histogram oluşturma
plt.figure(figsize=(12, 6))
plt.hist(company_data['Daily Return'], bins=20, alpha=0.7, label=tech_list[0])
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Daily Return for {tech_list[0]} (Past Year)')
plt.legend()
plt.tight_layout()
plt.show()


# 

# In[44]:


# NaN değerlerini kaldırma
df_cleaned = df.dropna()

# Pairplot for all numeric columns
sns.pairplot(df_cleaned)
plt.show()

# Autocorrelation Function (ACF)
plt.figure(figsize=(12, 6))
plot_acf(df_cleaned['Adj Close'], lags=320)
plt.title('ACF of Adj Close')
plt.show()

# Differenced Autocorrelation Function
plt.figure(figsize=(12, 6))
plot_acf(df_cleaned['Adj Close'].diff().dropna(), lags=40)
plt.title('Differenced ACF of Adj Close')
plt.show()

# Linear Regression
X = df_cleaned[['Adj Close']]
y = df_cleaned['Open']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, predictions, color='red', label='Linear Fit')
plt.xlabel('Adj Close')
plt.ylabel('Open')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Korelasyon matrisini hesaplamadan önce tarih sütununu çıkarın
numeric_data = df_cleaned.select_dtypes(include=[np.number])

# Correlation Matrix Heatmap
cor_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Korelasyon matrisini yazdırma
print("Correlation Matrix:")
print(cor_matrix)


# ## Interpreting the Correlation Matrix
# 
# The correlation matrix shows the relationship between each pair of numerical variables in your dataset. Here's how to interpret the results:
# 
# ### Correlation Coefficients:
# 
# - The correlation coefficient ranges from +1 to -1.
#   - +1: Perfect positive correlation (as one variable increases, the other also increases).
#   - -1: Perfect negative correlation (as one variable increases, the other decreases).
#   - 0: No correlation (no relationship between variables).
# 
# ### Correlation Matrix:
# 
# - **Open and High:** The correlation coefficient is 0.998878, indicating an almost perfect positive correlation. This means that as the Open price increases, the High price tends to increase as well.
# - **Open and Low:** The correlation coefficient is 0.998994, also indicating an almost perfect positive correlation.
# - **Open and Close/Adj Close:** The correlation coefficients are 0.999242 for both Close and Adj Close, showing a strong positive correlation.
# - **Volume with Others:** The correlation coefficients with Volume are negative and low (e.g., -0.149957 with Open). This indicates a weak negative relationship between volume and prices.
# 
# ### Overall Interpretation:
# 
# - **Strong Positive Correlation:** There are strong positive correlations between price variables (Open, High, Low, Close, Adj Close). This indicates that these variables tend to move together. For instance, if the opening price is high on a given day, the high, low, and closing prices are also generally high.
# - **Weak Negative Correlation:** There is a weak negative correlation between volume and price variables. This suggests that as trading volume increases, prices tend to decrease slightly, or vice versa, but this relationship is not strong.
# 
# Based on this correlation matrix, we can conclude that price variables are closely related to each other, while volume has a weak negative relationship with these prices. This information can guide further analyses or modeling efforts by highlighting which variables are most interrelated.
# 
# 
# 
# 
# 
# 

# 

# In[45]:


# Data cleaning

df_cleaned = df.dropna()

# Ensure the dataframe has the necessary number of rows for analysis
min_rows = df_cleaned.shape[0]
df_cleaned = df_cleaned.iloc[:min_rows]

# Print the cleaned dataframe to verify
print(df_cleaned.head())


# 

# In[48]:


from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plotting training and testing distributions
plt.figure(figsize=(12, 6))
plt.hist(y_train, bins=30, color='blue', alpha=0.7, label='y_train')
plt.title('Distribution of y_train')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(y_test, bins=30, color='red', alpha=0.7, label='y_test')
plt.title('Distribution of y_test')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[49]:


def prepare_data(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    df = stock_data.history(start=start_date, end=end_date)
    return df


# In[74]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import time
import matplotlib.dates as mdates
# Train and evaluate model function
def train_and_evaluate_model(df, company_name, ticker):
    # X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    # y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Regressor
    rf = RandomForestRegressor()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    
    start_time_rf = time.time()
    grid_search.fit(X_train_scaled, y_train)
    end_time_rf = time.time()
    
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test_scaled)

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()
    
    start_time_gbr = time.time()
    gbr.fit(X_train_scaled, y_train)
    end_time_gbr = time.time()
    
    y_pred_gbr = gbr.predict(X_test_scaled)

    # mse_rf = mean_squared_error(y_test, y_pred_rf)
    # mae_rf = mean_absolute_error(y_test, y_pred_rf)
    # mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    # mae_gbr = mean_absolute_error(y_test, y_pred_gbr)

    # print(f"{company_name} ({ticker}):")
    # print(f"Random Forest - MSE: {mse_rf}, MAE: {mae_rf}")
    # print(f"Gradient Boosting - MSE: {mse_gbr}, MAE: {mae_gbr}")

    # Plot Actual vs Predicted Prices
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', color='blue', alpha=0.7)
    # ax.plot(df.index[-len(y_test):], y_pred_rf, label='Predicted Prices (RF)', color='green', linestyle='--', alpha=0.7)
    # ax.plot(df.index[-len(y_test):], y_pred_gbr, label='Predicted Prices (GBR)', color='red', linestyle='--', alpha=0.7)
    # ax.legend()
    # ax.set_title(f'Actual vs Predicted Stock Prices for {company_name} ({ticker})')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Price')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
    
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

        # Metrics for Gradient Boosting
    mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    r2_gbr = r2_score(y_test, y_pred_gbr)
    mape_gbr = mean_absolute_percentage_error(y_test, y_pred_gbr)

    print(f"{company_name}:")
    print(f"Random Forest - MSE: {mse_rf}, MAE: {mae_rf}, R²: {r2_rf}, MAPE: {mape_rf}")
    print(f"Gradient Boosting - MSE: {mse_gbr}, MAE: {mae_gbr}, R²: {r2_gbr}, MAPE: {mape_gbr}")

    # Print training times
    print(f"Random Forest training time: {end_time_rf - start_time_rf} seconds")
    print(f"Gradient Boosting training time: {end_time_gbr - start_time_gbr} seconds")
    
    return y_test, y_pred_rf, y_pred_gbr

# Function to plot actual vs predicted prices
def plot_actual_vs_predicted(df, y_test, y_pred_rf, y_pred_gbr, company_name, ticker):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot Actual vs Predicted Prices (Random Forest)
    axs[0].plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', color='blue', alpha=0.7)
    axs[0].plot(df.index[-len(y_test):], y_pred_rf, label='Predicted Prices (RF)', color='green', linestyle='--', alpha=0.7)
    axs[0].legend()
    axs[0].set_title(f'Actual vs Predicted Stock Prices (Random Forest) for {company_name} ({ticker})')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[0].tick_params(axis='x', rotation=45)

    # Plot Actual vs Predicted Prices (Gradient Boosting)
    axs[1].plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', color='blue', alpha=0.7)
    axs[1].plot(df.index[-len(y_test):], y_pred_gbr, label='Predicted Prices (GBR)', color='red', linestyle='--', alpha=0.7)
    axs[1].legend()
    axs[1].set_title(f'Actual vs Predicted Stock Prices (Gradient Boosting) for {company_name} ({ticker})')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Modeli eğit ve değerlendir
y_test, y_pred_rf, y_pred_gbr = train_and_evaluate_model(df_cleaned, "NVIDIA", "NVDA")

# Plot actual vs predicted prices
plot_actual_vs_predicted(df_cleaned, y_test, y_pred_rf, y_pred_gbr, "NVIDIA", "NVDA")


# ### Evaluation of Model Results for NVIDIA
# 
# #### Model Performance Metrics:
# - **Random Forest:**
#   - **Mean Squared Error (MSE):** 0.3418326426914892
#   - **Mean Absolute Error (MAE):** 0.42458934293334627
#   - **R²:** 0.9926221875504637
#   - **Mean Absolute Percentage Error (MAPE):** 0.02780838834102856
# 
# - **Gradient Boosting:**
#   - **Mean Squared Error (MSE):** 0.3050026323469844
#   - **Mean Absolute Error (MAE):** 0.4008051535904705
#   - **R²:** 0.9934170938142328
#   - **Mean Absolute Percentage Error (MAPE):** 0.026271900815984406
# 
# #### Training Times:
# - **Random Forest training time:** 7.703564167022705 seconds
# - **Gradient Boosting training time:** 0.07957243919372559 seconds
# 
# #### Interpretation:
# 1. **Mean Squared Error (MSE):**
#    - Gradient Boosting has a lower MSE (0.305) compared to Random Forest (0.342), indicating that Gradient Boosting has better performance in terms of minimizing squared errors.
# 
# 2. **Mean Absolute Error (MAE):**
#    - Gradient Boosting has a lower MAE (0.401) compared to Random Forest (0.425), showing that Gradient Boosting performs better in terms of minimizing absolute errors.
# 
# 3. **R² (R-squared):**
#    - Both models have high R² values, indicating that they both explain a large proportion of the variance in the data. However, Gradient Boosting has a slightly higher R² (0.993) compared to Random Forest (0.993), suggesting it is slightly better at explaining the data variance.
# 
# 4. **Mean Absolute Percentage Error (MAPE):**
#    - Gradient Boosting has a lower MAPE (0.026) compared to Random Forest (0.028), indicating that Gradient Boosting has better performance in terms of minimizing percentage errors.
# 
# 5. **Training Time:**
#    - Gradient Boosting has a significantly shorter training time (0.080 seconds) compared to Random Forest (7.704 seconds). This is a substantial advantage, especially when working with large datasets.
# 
# ### Conclusion:
# - **Gradient Boosting** outperforms **Random Forest** in all metrics (MSE, MAE, R², and MAPE) and has a significantly shorter training time.
# - Based on these results, **Gradient Boosting** is the more suitable model for predicting NVIDIA stock prices.
# 
# 
# 
# 
# 
# 

# In[339]:


# def display_predicted_prices(selected_stock, df):
#     data = df.filter(['Close'])
#     dataset = data.values
#     training_data_len = int(np.ceil(len(dataset) * .95))
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(dataset)

#     train_data = scaled_data[:training_data_len, :]
#     x_train, y_train = [], []
#     for i in range(60, len(train_data)):
#         x_train.append(train_data[i-60:i, 0])
#         y_train.append(train_data[i, 0])
    
#     x_train, y_train = np.array(x_train), np.array(y_train)
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#     #
#     model = Sequential()
#     model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
#     model.add(LSTM(64, return_sequences=False))
#     model.add(Dense(25))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
     
#     history = model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
    
#     # Print the history keys and losses to verify training
#     print(f'{selected_stock} Training History:')
#     print(history.history.keys())
#     print(history.history['loss'])
    
#     test_data = scaled_data[training_data_len - 60:, :]
#     x_test = []
#     for i in range(60, len(test_data)):
#         x_test.append(test_data[i-60:i, 0])
#     x_test = np.array(x_test)
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#     predictions = model.predict(x_test)
#     predictions = scaler.inverse_transform(predictions)

#     # Generate prediction dates
#     prediction_dates = pd.date_range(end=df.index[-1], periods=len(predictions) + 1, freq='B')[1:]
    
#     # Plot with Plotly
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price', line=dict(color='cyan')))
#     fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price', line=dict(color='magenta')))
#     fig.update_layout(title=f'{selected_stock} Predicted Prices',
#                       xaxis_title='Date',
#                       yaxis_title='Price',
#                       plot_bgcolor='black',
#                       paper_bgcolor='black',
#                       font=dict(color='white'))
#     fig.show()
    
#     # Plot the training loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['loss'], label='Training Loss', color='cyan')
#     plt.title('LSTM Training Loss', color='white')
#     plt.xlabel('Epoch', color='white')
#     plt.ylabel('Loss', color='white')
#     plt.legend()
#     plt.show()

#     return df['Close'].iloc[-len(predictions):].values, predictions.flatten()

# # Run the prediction and display for each stock in company_list
# for company, name, ticker in zip(company_list, company_name, tech_list):
#     y_test_actual, predictions = display_predicted_prices(ticker, df_cleaned)

# # Assuming df is the last company's data
# df = company_list[-1]  # Adjust if needed
# y_test_actual, predictions = display_predicted_prices(tech_list[-1], df_cleaned)


# In[345]:


# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# def evaluate_lstm_performance(y_test_actual, y_test_pred):
#     mse_lstm = mean_squared_error(y_test_actual, y_test_pred)
#     mae_lstm = mean_absolute_error(y_test_actual, y_test_pred)
#     r2_lstm = r2_score(y_test_actual, y_test_pred)
#     mape_lstm = mean_absolute_percentage_error(y_test_actual, y_test_pred)
    
#     # Print metrics
#     print("LSTM Performance:")
#     print(f"MSE: {mse_lstm:.4f}")
#     print(f"MAE: {mae_lstm:.4f}")
#     print(f"R²: {r2_lstm:.4f}")
#     print(f"MAPE: {mape_lstm:.4f}")

# # Example usage
# evaluate_lstm_performance(y_test_actual, predictions)


# 

# 

# 

# In[344]:


# # Define the function for preparing the data
# def prepare_data(data, n_steps):
#     x, y = [], []
#     for i in range(len(data) - n_steps):
#         x.append(data[i:(i + n_steps), 0])
#         y.append(data[i + n_steps, 0])
#     return np.array(x), np.array(y)

# # Define the function to create and compile an LSTM model
# def create_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
#     model.add(LSTM(units=50))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Load the cleaned data
# # df_cleaned should be a pandas DataFrame with at least a 'Close' column
# # Example:
# # df_cleaned = pd.read_csv('path_to_your_cleaned_data.csv')

# # For demonstration purposes, we'll use dummy data:
# dates = pd.date_range(start='2015-01-01', periods=100, freq='D')
# closing_prices = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100)
# df_cleaned = pd.DataFrame({'Date': dates, 'Close': closing_prices})
# df_cleaned.set_index('Date', inplace=True)

# # Extract closing prices
# closing_prices = df_cleaned['Close'].values

# # Scale the closing prices
# scaler = MinMaxScaler(feature_range=(0, 1))
# closing_prices_scaled = scaler.fit_transform(closing_prices.reshape(-1, 1))

# # Prepare the training data
# n_steps = 60
# x_train, y_train = prepare_data(closing_prices_scaled, n_steps)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# # Create an instance of the LSTM model
# model = create_lstm_model((x_train.shape[1], 1))

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Generate predictions from the training set
# train_predictions = model.predict(x_train)
# train_predictions = scaler.inverse_transform(train_predictions)  # Reverse scaling

# # Get the actual closing prices for plotting
# actual_prices = scaler.inverse_transform(closing_prices_scaled)

# # Plot actual vs predicted prices
# plt.figure(figsize=(12, 6))
# plt.plot(df_cleaned.index[n_steps:], actual_prices[n_steps:], label='Actual Prices', color='blue')
# plt.plot(df_cleaned.index[n_steps:], train_predictions, label='Predicted Prices', color='red')
# plt.title('Stock Price Prediction using LSTM')
# plt.xlabel('Date')
# plt.ylabel('Stock Price (USD)')
# plt.legend()
# plt.show()


# In[343]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense
# from sklearn.metrics import mean_absolute_error as mae

# # Define the function for preparing the data
# def prepare_data(data, n_steps):
#     x, y = [], []
#     for i in range(len(data) - n_steps):
#         x.append(data[i:(i + n_steps), 0])
#         y.append(data[i + n_steps, 0])
#     return np.array(x), np.array(y)

# # For demonstration purposes, we'll use dummy data:
# dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
# closing_prices = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100)
# df_cleaned = pd.DataFrame({'Date': dates, 'Close': closing_prices})
# df_cleaned.set_index('Date', inplace=True)

# # Extract and scale the closing prices
# closing_prices = df_cleaned['Close'].values

# # MinMax Scaling
# min_max_scaler = MinMaxScaler(feature_range=(0, 1))
# closing_prices_scaled = min_max_scaler.fit_transform(closing_prices.reshape(-1, 1))

# # Determine appropriate n_past value based on the length of the data
# data_length = len(closing_prices_scaled)
# n_past = 60  # Set your desired n_past value

# # Check if the split is large enough
# if data_length <= n_past:
#     print(f"Warning: Insufficient data to create TimeseriesGenerator with the given n_past ({n_past}).")
#     print(f"Total number of samples: {data_length}")
#     # You can either reduce n_past or proceed with the available data
#     # For example, reducing n_past to the minimum length of available data:
#     n_past = data_length - 1
#     print(f"Adjusting n_past to: {n_past}")

# # Prepare the data for LSTM
# x, y = prepare_data(closing_prices_scaled, n_past)
# x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# # Split the data into training and testing sets
# split = int(len(x) * 0.8)
# x_train, x_test = x[:split], x[split:]
# y_train, y_test = y[:split], y[split:]

# # Check if the split is large enough
# print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")
# if len(x_train) <= n_past or len(x_test) <= n_past:
#     print(f"Warning: Even after adjusting n_past, the data size may still be insufficient.")
#     print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")

# # Define and compile the LSTM model
# num_feature = 1
# model = Sequential()
# model.add(LSTM(500, activation='tanh', input_shape=(n_past, num_feature), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(400, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(200, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(100, return_sequences=False))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# # Train the model directly on the prepared data
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), shuffle=False, batch_size=20, verbose=1)

# # Plot training and validation loss
# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.title('LSTM Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# # Generate predictions
# predictions = model.predict(x_test)

# # Reverse scaling of predictions
# predictions = min_max_scaler.inverse_transform(predictions)

# # Prepare data for plotting
# df_pred = pd.DataFrame(predictions, columns=['Predicted'])
# df_pred.index = df_cleaned.index[-len(predictions):]
# df_final = df_cleaned[['Close']].iloc[-len(predictions):].copy()
# df_final['Predicted'] = df_pred['Predicted']

# # Plot actual vs predicted values
# plt.figure(figsize=(15, 12))
# plt.plot(df_final['Close'], label='Actual Prices')
# plt.plot(df_final['Predicted'], label='Predicted Prices')
# plt.legend(loc="upper right")
# plt.title('LSTM Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.show()

# # Calculate RMSE and MAE
# rmse = np.sqrt(np.mean((df_final['Predicted'] - df_final['Close'])**2))
# mae_value = mae(df_final['Predicted'], df_final['Close'])

# print(f"Root Mean Square Error (RMSE): {rmse}")
# print(f"Mean Absolute Error (MAE): {mae_value}")


# In[331]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import RobustScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense
# from sklearn.metrics import mean_absolute_error as mae

# # Define the function for preparing the data
# def prepare_data(data, n_steps):
#     x, y = [], []
#     for i in range(len(data) - n_steps):
#         x.append(data[i:(i + n_steps), 0])
#         y.append(data[i + n_steps, 0])
#     return np.array(x), np.array(y)

# # Create a date range from January 1, 2015 to December 31, 2025
# dates = pd.date_range(start='2024-05-01', periods=100, freq='D')

# # Generate closing prices with the correct length
# closing_prices = np.sin(np.linspace(0, 20, len(dates))) + np.random.normal(0, 0.1, len(dates))

# # Create a DataFrame with the generated data
# df_cleaned = pd.DataFrame({'Date': dates, 'Close': closing_prices})
# df_cleaned.set_index('Date', inplace=True)

# # Extract and scale the closing prices
# closing_prices = df_cleaned['Close'].values

# # Robust Scaling
# robust_scaler = RobustScaler()
# closing_prices_scaled = robust_scaler.fit_transform(closing_prices.reshape(-1, 1))

# # Determine appropriate n_past value based on the length of the data
# data_length = len(closing_prices_scaled)
# n_past = 60  # Set your desired n_past value

# # Check if the split is large enough
# if data_length <= n_past:
#     print(f"Warning: Insufficient data to create TimeseriesGenerator with the given n_past ({n_past}).")
#     print(f"Total number of samples: {data_length}")
#     # You can either reduce n_past or proceed with the available data
#     # For example, reducing n_past to the minimum length of available data:
#     n_past = data_length - 1
#     print(f"Adjusting n_past to: {n_past}")

# # Prepare the data for LSTM
# x, y = prepare_data(closing_prices_scaled, n_past)
# x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# # Split the data into training and testing sets
# split = int(len(x) * 0.8)
# x_train, x_test = x[:split], x[split:]
# y_train, y_test = y[:split], y[split:]

# # Check if the split is large enough
# print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")
# if len(x_train) <= n_past or len(x_test) <= n_past:
#     print(f"Warning: Even after adjusting n_past, the data size may still be insufficient.")
#     print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")

# # Define and compile the LSTM model
# num_feature = 1
# model = Sequential()
# model.add(LSTM(500, activation='tanh', input_shape=(n_past, num_feature), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(400, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(200, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(100, return_sequences=False))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# # Train the model directly on the prepared data
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), shuffle=False, batch_size=20, verbose=1)

# # Plot training and validation loss
# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.title('LSTM Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# # Generate predictions
# predictions = model.predict(x_test)

# # Reverse scaling of predictions
# predictions = robust_scaler.inverse_transform(predictions)

# # Prepare data for plotting
# df_pred = pd.DataFrame(predictions, columns=['Predicted'])
# df_pred.index = df_cleaned.index[-len(predictions):]
# df_final = df_cleaned[['Close']].iloc[-len(predictions):].copy()
# df_final['Predicted'] = df_pred['Predicted']

# # Plot actual vs predicted values
# plt.figure(figsize=(15, 12))
# plt.plot(df_final['Close'], label='Actual Prices')
# plt.plot(df_final['Predicted'], label='Predicted Prices')
# plt.legend(loc="upper right")
# plt.title('LSTM Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.show()

# # Calculate RMSE and MAE
# rmse = np.sqrt(np.mean((df_final['Predicted'] - df_final['Close'])**2))
# mae_value = mae(df_final['Predicted'], df_final['Close'])

# print(f"Root Mean Square Error (RMSE): {rmse}")
# print(f"Mean Absolute Error (MAE): {mae_value}")


# In[336]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_absolute_error as mae

# Define the function for preparing the data
def prepare_data(data, n_steps):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(x), np.array(y)

# Create a date range from July 1, 2024 to September 30, 2024
future_dates = pd.date_range(start='2024-05-01', periods= 100, freq='D')

# Generate closing prices with the correct length
closing_prices = np.sin(np.linspace(0, 20, len(future_dates))) + np.random.normal(0, 0.1, len(future_dates))

# Create a DataFrame with the generated data
df_cleaned = pd.DataFrame({'Date': future_dates, 'Close': closing_prices})
df_cleaned.set_index('Date', inplace=True)

# Extract and scale the closing prices
closing_prices = df_cleaned['Close'].values.reshape(-1, 1)

# Robust Scaling
robust_scaler = RobustScaler()
closing_prices_scaled = robust_scaler.fit_transform(closing_prices)

# Determine appropriate n_past value based on the length of the data
data_length = len(closing_prices_scaled)
n_past = 60  # Set your desired n_past value

# Check if the split is large enough
if data_length <= n_past:
    print(f"Warning: Insufficient data to create TimeseriesGenerator with the given n_past ({n_past}).")
    print(f"Total number of samples: {data_length}")
    # You can either reduce n_past or proceed with the available data
    n_past = data_length - 1
    print(f"Adjusting n_past to: {n_past}")

# Prepare the data for LSTM
x, y = prepare_data(closing_prices_scaled, n_past)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Split the data into training and testing sets
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# Check if the split is large enough
print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")
if len(x_train) <= n_past or len(x_test) <= n_past:
    print(f"Warning: Even after adjusting n_past, the data size may still be insufficient.")
    print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")

# Define and compile the LSTM model
num_feature = 1
model = Sequential()
model.add(LSTM(500, activation='tanh', input_shape=(n_past, num_feature), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Track training start time
start_time_lstm = time.time()
# Train the model directly on the prepared data
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), shuffle=False, batch_size=20, verbose=1)
# Track training end time
end_time_lstm = time.time()
# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Generate predictions
predictions = model.predict(x_test)

# Reverse scaling of predictions
predictions = robust_scaler.inverse_transform(predictions)

# Prepare data for plotting
df_pred = pd.DataFrame(predictions, columns=['Predicted'])
df_pred.index = df_cleaned.index[-len(predictions):]
df_final = df_cleaned[['Close']].iloc[-len(predictions):].copy()
df_final['Predicted'] = df_pred['Predicted']

# Plot actual vs predicted values
plt.figure(figsize=(15, 12))
plt.plot(df_final['Close'], label='Actual Prices')
plt.plot(df_final['Predicted'], label='Predicted Prices')
plt.legend(loc="upper right")
plt.title('LSTM Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# Calculate RMSE and MAE
rmse = np.sqrt(np.mean((df_final['Predicted'] - df_final['Close'])**2))
mae_value = mae(df_final['Predicted'], df_final['Close'])

print(f"Root Mean Square Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae_value}")


# Calculate RMSE and MAE for LSTM
rmse_lstm = np.sqrt(mean_squared_error(df_final['Close'], df_final['Predicted']))
mae_lstm = mean_absolute_error(df_final['Close'], df_final['Predicted'])
r2_lstm = r2_score(df_final['Close'], df_final['Predicted'])
mape_lstm = mean_absolute_percentage_error(df_final['Close'], df_final['Predicted'])

print(f"LSTM Model:")
print(f"Mean Squared Error (MSE): {rmse_lstm**2}")
print(f"Mean Absolute Error (MAE): {mae_lstm}")
print(f"R² Score: {r2_lstm}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_lstm}")

# Print training times
print(f"LSTM training time: {end_time_lstm - start_time_lstm} seconds")


# In[342]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import RobustScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense
# from sklearn.metrics import mean_absolute_error as mae

# # Define the function for preparing the data
# def prepare_data(data, n_steps):
#     x, y = [], []
#     for i in range(len(data) - n_steps):
#         x.append(data[i:(i + n_steps), 0])
#         y.append(data[i + n_steps, 0])
#     return np.array(x), np.array(y)

# # Create a date range from July 1, 2024 to September 30, 2024
# future_dates = pd.date_range(start='2020-05-01', end='2024-10-01', freq='D')

# # Generate closing prices with the correct length
# closing_prices = np.sin(np.linspace(0, 20, len(future_dates))) + np.random.normal(0, 0.1, len(future_dates))

# # Create a DataFrame with the generated data
# df_cleaned = pd.DataFrame({'Date': future_dates, 'Close': closing_prices})
# df_cleaned.set_index('Date', inplace=True)

# # Extract and scale the closing prices
# closing_prices = df_cleaned['Close'].values.reshape(-1, 1)

# # Robust Scaling
# robust_scaler = RobustScaler()
# closing_prices_scaled = robust_scaler.fit_transform(closing_prices)

# # Determine appropriate n_past value based on the length of the data
# data_length = len(closing_prices_scaled)
# n_past = 60  # Set your desired n_past value

# # Check if the split is large enough
# if data_length <= n_past:
#     print(f"Warning: Insufficient data to create TimeseriesGenerator with the given n_past ({n_past}).")
#     print(f"Total number of samples: {data_length}")
#     # You can either reduce n_past or proceed with the available data
#     n_past = data_length - 1
#     print(f"Adjusting n_past to: {n_past}")

# # Prepare the data for LSTM
# x, y = prepare_data(closing_prices_scaled, n_past)
# x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# # Split the data into training and testing sets
# split = int(len(x) * 0.8)
# x_train, x_test = x[:split], x[split:]
# y_train, y_test = y[:split], y[split:]

# # Check if the split is large enough
# print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")
# if len(x_train) <= n_past or len(x_test) <= n_past:
#     print(f"Warning: Even after adjusting n_past, the data size may still be insufficient.")
#     print(f"Length of x_train: {len(x_train)}, Length of x_test: {len(x_test)}")

# # Define and compile the LSTM model
# num_feature = 1
# model = Sequential()
# model.add(LSTM(500, activation='tanh', input_shape=(n_past, num_feature), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(400, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(200, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(100, return_sequences=False))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# # Track training start time
# start_time_lstm = time.time()
# # Train the model directly on the prepared data
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), shuffle=False, batch_size=10, verbose=1)
# # Track training end time
# end_time_lstm = time.time()
# # Plot training and validation loss
# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.title('LSTM Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# # Generate predictions
# predictions = model.predict(x_test)

# # Reverse scaling of predictions
# predictions = robust_scaler.inverse_transform(predictions)

# # Prepare data for plotting
# df_pred = pd.DataFrame(predictions, columns=['Predicted'])
# df_pred.index = df_cleaned.index[-len(predictions):]
# df_final = df_cleaned[['Close']].iloc[-len(predictions):].copy()
# df_final['Predicted'] = df_pred['Predicted']

# # Plot actual vs predicted values
# plt.figure(figsize=(15, 12))
# plt.plot(df_final['Close'], label='Actual Prices')
# plt.plot(df_final['Predicted'], label='Predicted Prices')
# plt.legend(loc="upper right")
# plt.title('LSTM Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.show()

# # Calculate RMSE and MAE
# rmse = np.sqrt(np.mean((df_final['Predicted'] - df_final['Close'])**2))
# mae_value = mae(df_final['Predicted'], df_final['Close'])

# print(f"Root Mean Square Error (RMSE): {rmse}")
# print(f"Mean Absolute Error (MAE): {mae_value}")


# # Calculate RMSE and MAE for LSTM
# rmse_lstm = np.sqrt(mean_squared_error(df_final['Close'], df_final['Predicted']))
# mae_lstm = mean_absolute_error(df_final['Close'], df_final['Predicted'])
# r2_lstm = r2_score(df_final['Close'], df_final['Predicted'])
# mape_lstm = mean_absolute_percentage_error(df_final['Close'], df_final['Predicted'])

# print(f"LSTM Model:")
# print(f"Mean Squared Error (MSE): {rmse_lstm**2}")
# print(f"Mean Absolute Error (MAE): {mae_lstm}")
# print(f"R² Score: {r2_lstm}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape_lstm}")

# # Print training times
# print(f"LSTM training time: {end_time_lstm - start_time_lstm} seconds")


# ### Evaluation of Model Results for NVIDIA
# 
# #### Model Performance Metrics:
# 
# **1. Random Forest:**
# - **Mean Squared Error (MSE):** 0.3418
# - **Mean Absolute Error (MAE):** 0.4246
# - **R² Score:** 0.9926
# - **Mean Absolute Percentage Error (MAPE):** 0.0278
# 
# **2. Gradient Boosting:**
# - **Mean Squared Error (MSE):** 0.3050
# - **Mean Absolute Error (MAE):** 0.4008
# - **R² Score:** 0.9934
# - **Mean Absolute Percentage Error (MAPE):** 0.0263
# 
# **3. LSTM Model:**
# - **Mean Squared Error (MSE):** 0.0181
# - **Mean Absolute Error (MAE):** 0.1131
# - **R² Score:** 0.9108
# - **Mean Absolute Percentage Error (MAPE):** 1.8978
# - **Training Time:** 15.6092 seconds
# 
# ### Comparative Analysis
# 
# 1. **Accuracy Metrics:**
#    - **MSE & MAE:** The LSTM model exhibits significantly lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) compared to the Random Forest and Gradient Boosting models, indicating it predicts stock prices with less error.
#    - **R² Score:** The R² Score for LSTM (0.9108) is lower than that of Random Forest (0.9926) and Gradient Boosting (0.9934), suggesting that LSTM explains a smaller proportion of the variance in the data.
#    - **MAPE:** The Mean Absolute Percentage Error (MAPE) for LSTM is much higher (1.8978) compared to Random Forest (0.0278) and Gradient Boosting (0.0263), indicating less reliability in percentage error terms.
# 
# 2. **Training Times:**
#    - **LSTM Training Time:** The training time for the LSTM model (15.6092 seconds) is significantly higher compared to Random Forest (7.7036 seconds) and Gradient Boosting (0.0796 seconds), implying that LSTM takes more time to train.
# 
# ### Summary:
# 
# - **LSTM:** While it has lower MSE and MAE, suggesting better prediction accuracy, it requires more training time and has a lower R² score compared to Random Forest and Gradient Boosting. Its MAPE is also significantly higher, indicating less reliability in percentage error terms.
# 
# - **Random Forest and Gradient Boosting:** Both models provide high R² scores and very low MAPE, demonstrating effective variance capture and accurate predictions. They also train much faster than LSTM.
# 
# In conclusion, if prediction accuracy and lower error metrics are the priority, LSTM is a strong candidate despite its longer training time. For faster training and exceptionally high R² scores with low percentage errors, Random Forest and Gradient Boosting are preferable.

# 4. Visualising stock data
# Japanese candlestick charts are tools used in a particular trading style called price action to predict market movement through pattern recognition of continuations, breakouts and reversals.
# 
# Unlike a line chart, all of the price information can be viewed in one figure showing the high, low, open and close price of the day or chosen time frame. Price action traders observe patterns formed by green bullish candles where the stock is trending upwards over time, and red or black bearish candles where there is a downward trend.

# In[181]:


def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
    """
    Japanese candlestick chart showing OHLC prices for a specified time period
    
    :param dat: pandas dataframe object with datetime64 index, and float columns "Open", "High", "Low", and "Close"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
 
    :returns: a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
 
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "green", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    sns.set(rc={'figure.figsize':(20, 10)})
    plt.style.use('seaborn-whitegrid')
    plt.title(f"Candlestick chart of {txt}", color = 'black', fontsize = 20)
    plt.xlabel('Date', color = 'black', fontsize = 15)
    plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
 
    plt.show()


# In[199]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime

# Ensure matplotlib inline plots in Jupyter Notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Sample data - replace this with your actual data loading code
# Example data loading for demonstration purposes
start_date = '2023-01-01'
end_date = '2024-12-31'
ticker = 'NVDA'
data = yf.Ticker(ticker)
df = data.history(start=start_date, end=end_date)

# Reset index to have Date as a column
df.reset_index(inplace=True)

# Ensure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# Prepare the data for mplfinance
mpf_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Plot candlestick chart
mpf.plot(mpf_data, type='candle', style='charles', title=f'{ticker} Stock Prices from {start_date} - {end_date}', ylabel='Price', volume=True)


# Technical Indicators and Strategies
# A technical indicator is a series of data points that are derived by applying a formula to the price data of a security. Basically, they are price-derived indicators that use formulas to translate the momentum or price levels into quantifiable time series.
# 
# There are two categories of indicator: leading and lagging, and four types: trend, momentum, volatility and volume, which serve three broad functions: to alert, to confirm and to predict

# Trend-following strategies
# Trend-following is about profiting from the prevailing trend through buying an asset when its price trend goes up, and selling when its trend goes down, expecting price movements to continue.
# 
# Moving averages
# Moving averages smooth a series filtering out noise to help identify trends, one of the fundamental principles of technical analysis being that prices move in trends. Types of moving averages include simple, exponential, smoothed, linear-weighted, MACD, and as lagging indicators they follow the price action and are commonly referred to as trend-following indicators.
# 
# Simple Moving Average (SMA)
# The simplest form of a moving average, known as a Simple Moving Average (SMA), is calculated by taking the arithmetic mean of a given set of values over a set time period. This model is probably the most naive approach to time series modelling and simply states that the next observation is the mean of all past observations and each value in the time period carries equal weight.
# 
# Modelling this an as average calculation problem we would try to predict the future stock market prices (for example, xt+1 ) as an average of the previously observed stock market prices within a fixed size window (for example, xt-n, ..., xt). This helps smooth out the price data by creating a constantly updated average price so that the impacts of random, short-term fluctuations on the price of a stock over a specified time-frame are mitigated.

# In[212]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set ticker and title
ticker = 'NVDA'
title_txt = "20-day Simple Moving Average for NVDA stock"
label_txt = "NVDA Adj Close"

# Load/Read Data
yf.pdr_override()

# Define company tickers
tech_list = ['NVDA']

# Download stock data for the past 7 years
end = datetime.now()
start = datetime(end.year - 7, end.month, end.day)

# Initialize empty DataFrame
df = pd.DataFrame()

# Download and concatenate stock data
for stock in tech_list:
    temp_df = yf.download(stock, start=start, end=end)
    temp_df['Ticker'] = stock
    df = pd.concat([df, temp_df])

# Reset index to make 'Date' a column
df.reset_index(inplace=True)

# Print the column names to verify
print("Columns in DataFrame:", df.columns)

# If 'Adj Close' exists, calculate the 20-day SMA
if 'Adj Close' in df.columns:
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()

    # Plot the adjusted close and the SMA
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Adj Close'], label='Adj Close', color='blue')
    plt.plot(df['Date'], df['SMA_20'], label='20-Day SMA', color='red')
    plt.title(title_txt)
    plt.xlabel('Date')
    plt.ylabel(label_txt)
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Column 'Adj Close' not found in DataFrame.")


# The SMA follows the time series removing noise from the signal and keeping the relevant information about the trend. If the stock price is above its moving average it is assumed that it will likely continue rising in an uptrend.
# 
# Moving Average Crossover Strategy
# The most popular moving average crossover strategy, and the "Hello World!" of quantitative trading, being the easiest to construct, is based on the simple moving average. When moving averages cross, it is usually confirmation of a change in the prevailing trend, and we want to test whether over the long term the lag caused by the moving average can still give us profitable trades.
# 
# Depending on the type of investor or trader (high risk vs. low risk, short-term vs. long-term trading), you can adjust your moving ‘time’ average (10 days, 20 days, 50 days, 200 days, 1 year, 5 years, etc). The longer the period of an SMA, the longer the time horizon of the trend it spots. The most commonly used SMA periods are 20 for short-term (swing) trading, 50 for medium-term (position) trading and 200 for long-term (portfolio) trading.
# 
# There is no single right answer and this will vary according to whether a trader is planning to buy when the trend is going down and sell when it's going up, potentially making short-term gains, or to hold for a more long-term investment.



import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set ticker and title
ticker = 'NVDA'
title_txt = "20, 50, and 200-day Moving Averages for NVDA Stock"
label_txt = "NVDA Adj Close"

# Load/Read Data
yf.pdr_override()

# Define company tickers
tech_list = [ticker]

# Download stock data for the past 7 years
end = datetime.now()
start = datetime(end.year - 7, end.month, end.day)

# Initialize empty DataFrame
df = pd.DataFrame()

# Download and concatenate stock data
for stock in tech_list:
    temp_df = yf.download(stock, start=start, end=end)
    temp_df['Ticker'] = stock
    df = pd.concat([df, temp_df])

# Reset index to make 'Date' a column
df.reset_index(inplace=True)

# Print the column names to verify
print("Columns in DataFrame:", df.columns)

def sma2():
    plt.figure(figsize=(15,9))
    # Calculate moving averages
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
    
    # Plot moving averages
    plt.plot(df['Date'], df['SMA_20'], label='20 Day Avg', color='orange')
    plt.plot(df['Date'], df['SMA_50'], label='50 Day Avg', color='green')
    plt.plot(df['Date'], df['SMA_200'], label='200 Day Avg', color='blue')
    plt.plot(df['Date'], df['Adj Close'], label=label_txt, color='black')
    
    plt.title(title_txt, color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (USD)', color='black', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

sma2()


# The chart shows that the 20-day moving average is the most sensitive to local changes, and the 200-day moving average the least. Here, the 200-day moving average indicates an overall bullish trend - the stock is trending upward over time. The 20- and 50-day moving averages are at times bearish and at other times bullish.
# 
# The major drawback of moving averages, however, is that because they are lagging, and smooth out prices, they tend to recognise reversals too late and are therefore not very helpful when used alone.

# Trading Strategy
# The moving average crossover trading strategy will be to take two moving averages - 20-day (fast) and 200-day (slow) - and to go long (buy) when the fast MA goes above the slow MA and to go short (sell) when the fast MA goes below the slow MA.




# Create copy of dataframe for AstraZeneca data for 2010-2019

nvda_sma = temp_df.copy()
     

nvda_sma


# In[218]:


# Calculate and add columns for moving averages of Adjusted Close price data

nvda_sma["20d"] = np.round(nvda_sma["Adj Close"].rolling(window = 20, center = False).mean(), 2)
nvda_sma["50d"] = np.round(nvda_sma["Adj Close"].rolling(window = 50, center = False).mean(), 2)
nvda_sma["200d"] = np.round(nvda_sma["Adj Close"].rolling(window = 200, center = False).mean(), 2)

nvda_sma.tail()


# In[219]:


txt = "20, 50 and 200 day moving averages for NVDA stock"

# Slice rows to plot data from 2018-2024
pandas_candlestick_ohlc(nvda_sma.loc['2018-01-01':'2024-12-31',:], otherseries = ["20d", "50d", "200d"])


# Backtesting
# Before using the strategy we will evaluate the quality of it first by backtesting, or looking at how profitable it is on historical data.

# In[221]:


# Identify when the 20-day average is below the 200-day average, and vice versa.

nvda_sma['20d-200d'] = nvda_sma['20d'] - nvda_sma['200d']
nvda_sma.tail()


# In[223]:


# The sign of this difference is the regime; that is, if the fast moving average is above the slow moving average, 
# this is a bullish regime, and a bearish regime holds when the fast moving average is below the slow moving average

# np.where() is a vectorized if-else function, where a condition is checked for each component of a vector, and the first argument passed is used when the condition holds, and the other passed if it does not
nvda_sma["Regime"] = np.where(nvda_sma['20d-200d'] > 0, 1, 0)
# We have 1's for bullish regimes and 0's for everything else. Replace bearish regime's values with -1, and to maintain the rest of the vector, the second argument is nvda_sma["Regime"]
nvda_sma["Regime"] = np.where(nvda_sma['20d-200d'] < 0, -1, nvda_sma["Regime"])
nvda_sma.loc['2018-01-01':'2024-12-31',"Regime"].plot(ylim = (-2,2)).axhline(y = 0, color = "black", lw = 2);
plt.title("Regime for NVDA 20- and 200-day Moving Average Crossover Strategy for 2018-2024", color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Regime', color = 'black', fontsize = 15);


# In[224]:


nvda_sma["Regime"].plot(ylim = (-2,2)).axhline(y = 0, color = "black", lw = 2);
plt.title("Regime for NVDA 20- and 200-day Moving Average Crossover Strategy for 2018-2024", color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Regime', color = 'black', fontsize = 15);


# Number of bullish and bearish days

nvda_sma["Regime"].value_counts()


# For 1172 days the market was bullish, for 388 days it was bearish, and neutral for 200 days for the time period 2018-2024.

# In[226]:


nvda_sma


# In[233]:


# Obtain signals with -1 indicating “sell”, 1 indicating “buy”, and 0 no action
# To ensure that all trades close out, temporarily change the regime of the last row to 0
regime_orig = nvda_sma.iloc[-1, 10]
nvda_sma.iloc[-1, 10] = 0
nvda_sma["Signal"] = np.sign(nvda_sma["Regime"] - nvda_sma["Regime"].shift(1))
# Restore original regime data
nvda_sma.iloc[-1, 10] = regime_orig
nvda_sma.tail()


# In[234]:


nvda_sma["Signal"].plot(ylim = (-2, 2));
plt.title("Trading signals for NVDA 20- and 200-day Moving Average Crossover Strategy for 2018-2024", color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Trading signal', color = 'black', fontsize = 15);


# In[235]:


# Unique counts of trading signals

nvda_sma["Signal"].value_counts()


# We would buy NVDA stock 6 times and sell 4 times. If we only go long 6 trades will be engaged in over the 6-year period, while if we pivot from a long to a short position every time a long position is terminated, we would engage in 6 trades total. It is worth bearing in mind that trading more frequently isn’t necessarily good as trades are never free.

# In[237]:


# Identify what the price of the stock is at every buy.

nvda_sma.loc[nvda_sma["Signal"] == 1, "Close"]


# In[238]:


# Identify what the price of the stock is at every sell.

nvda_sma.loc[nvda_sma["Signal"] == -1, "Close"]


# In[239]:


# Create a dataframe with trades, including the price at the trade and the regime under which the trade is made.

nvda_signals = pd.concat([
        pd.DataFrame({"Price": nvda_sma.loc[nvda_sma["Signal"] == 1, "Adj Close"],
                     "Regime": nvda_sma.loc[nvda_sma["Signal"] == 1, "Regime"],
                     "Signal": "Buy"}),
        pd.DataFrame({"Price": nvda_sma.loc[nvda_sma["Signal"] == -1, "Adj Close"],
                     "Regime": nvda_sma.loc[nvda_sma["Signal"] == -1, "Regime"],
                     "Signal": "Sell"}),
    ])
nvda_signals.sort_index(inplace = True)
nvda_signals


# In[250]:


# Ensure previous_buy_signals aligns with buy_signals
buy_signals = nvda_signals[nvda_signals['Signal'] == 'Buy']
previous_buy_signals = buy_signals.shift(1)

# Create DataFrame for long trade profits
nvda_long_profits = pd.DataFrame({
    "Entry Price": buy_signals["Price"],
    "Previous Buy Price": previous_buy_signals["Price"].values,
    "Profit": buy_signals["Price"].values - previous_buy_signals["Price"].values,
    "End Date": buy_signals.index
}).dropna()  # Drop rows with NaN values in 'Profit'

# Print the nvda_long_profits DataFrame
print("Columns in nvda_long_profits:", nvda_long_profits.columns)
print(nvda_long_profits)


#Exponential Moving Average
#In a Simple Moving Average, each value in the time period carries equal weight, and values outside of the time period are not included in the average. However, the Exponential Moving Average is a cumulative calculation where a different decreasing weight is assigned to each observation. Past values have a diminishing contribution to the average, while more recent values have a greater contribution. This method allows the moving average to be more responsive to changes in the data.

# In[256]:


# Set ticker and title
ticker = 'NVDA'
title_txt = "20-day Exponential Moving Average for NVDA stock"
label_txt = "NVDA Adj Close"

# Download NVDA stock data for the year 2024
end = datetime(2024, 12, 31)
start = datetime(2024, 1, 1)

# Download stock data for the specified ticker
df = yf.download(ticker, start=start, end=end)

# Define ewma function
def ewma():
    plt.figure(figsize=(15, 9))
    # Calculate and plot 20-day EMA
    df['20_Day_EMA'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
    df['Adj Close'].plot(label=label_txt, color='blue')
    df['20_Day_EMA'].plot(label='20 Day EMA', color='red')
    
    plt.title(title_txt, color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot
ewma()


# In[257]:


# Set ticker and title
ticker = 'NVDA'
title_txt = "20-, 50-, and 200-day Exponential Moving Averages for NVDA stock"
label_txt = "NVDA Adj Close"

# Download NVDA stock data for the period 2016-2019
start = datetime(2018, 1, 1)
end = datetime(2024, 12, 31)

# Download stock data for the specified ticker
df = yf.download(ticker, start=start, end=end)

# Define ewma2 function
def ewma2():
    plt.figure(figsize=(15, 9))
    # Calculate and plot 20-day EMA
    df['20_Day_EMA'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
    df['50_Day_EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
    df['200_Day_EMA'] = df['Adj Close'].ewm(span=200, adjust=False).mean()
    
    # Plot adjusted close and EMAs
    df['Adj Close'].plot(label=label_txt, color='blue')
    df['20_Day_EMA'].plot(label='20 Day EMA', color='red')
    df['50_Day_EMA'].plot(label='50 Day EMA', color='green')
    df['200_Day_EMA'].plot(label='200 Day EMA', color='orange')
    
    plt.title(title_txt, color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot
ewma2()


# Triple Moving Average Crossover Strategy
# This strategy uses three moving moving averages - short/fast, middle/medium and long/slow - and has two buy and sell signals.
# 
# The first is to buy when the middle/medium moving average crosses above the long/slow moving average and the short/fast moving average crosses above the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses below the middle/medium moving average.
# 
# The second is to buy when the middle/medium moving average crosses below the long/slow moving average and the short/fast moving average crosses below the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses above the middle/medium moving average.

# In[259]:


nvda_sma[['Adj Close']]['2024-05-01':'2024-10-31']


# In[260]:


# Identify what the price of the stock is at every sell.

nvda_sma.loc[nvda_sma["Signal"] == -1, "Close"]


# In[264]:


# Create a dataframe with trades, including the price at the trade and the regime under which the trade is made.

nvda_signals = pd.concat([
        pd.DataFrame({"Price": nvda_sma.loc[nvda_sma["Signal"] == 1, "Adj Close"],
                     "Regime": nvda_sma.loc[nvda_sma["Signal"] == 1, "Regime"],
                     "Signal": "Buy"}),
        pd.DataFrame({"Price": nvda_sma.loc[nvda_sma["Signal"] == -1, "Adj Close"],
                     "Regime": nvda_sma.loc[nvda_sma["Signal"] == -1, "Regime"],
                     "Signal": "Sell"}),
    ])
nvda_signals.sort_index(inplace = True)
nvda_signals


# In[267]:


# Let's see the profitability of long trades

# Create DataFrame for long trade profits
nvda_long_profits = pd.DataFrame({
    "Price": buy_signals["Price"],
    "Previous Buy Price": previous_buy_signals["Price"].reindex(buy_signals.index).values,
    "Profit": buy_signals["Price"].values - previous_buy_signals["Price"].reindex(buy_signals.index).values,
    "End Date": buy_signals.index
}).dropna()  # Drop rows with NaN values in 'Profit'

# Print the nvda_long_profits DataFrame
print(nvda_long_profits)


# Triple Moving Average Crossover Strategy
# This strategy uses three moving moving averages - short/fast, middle/medium and long/slow - and has two buy and sell signals.
# 
# The first is to buy when the middle/medium moving average crosses above the long/slow moving average and the short/fast moving average crosses above the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses below the middle/medium moving average.
# 
# The second is to buy when the middle/medium moving average crosses below the long/slow moving average and the short/fast moving average crosses below the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses above the middle/medium moving average.

# In[282]:


# Define the function for plotting EMAs
def ewma3():
    sns.set(rc={'figure.figsize':(15, 9)})
    
    # Extract data for the 6-month period
    nvda_adj_6mo1 = nvda_sma[['Adj Close']]['2024-05-01':'2024-10-31']
    
    # Calculate EMAs
    ShortEMA = nvda_adj_6mo1['Adj Close'].ewm(span=5, adjust=False).mean()
    MiddleEMA = nvda_adj_6mo1['Adj Close'].ewm(span=21, adjust=False).mean()
    LongEMA = nvda_adj_6mo1['Adj Close'].ewm(span=63, adjust=False).mean()
    
    # Add EMAs to the DataFrame
    nvda_adj_6mo1['Short'] = ShortEMA
    nvda_adj_6mo1['Middle'] = MiddleEMA
    nvda_adj_6mo1['Long'] = LongEMA
    
    # Plotting
    plt.plot(nvda_adj_6mo1['Adj Close'], label=f"{label_txt}", color='blue')
    plt.plot(ShortEMA, label='Short/Fast EMA', color='red')
    plt.plot(MiddleEMA, label='Middle/Medium EMA', color='orange')
    plt.plot(LongEMA, label='Long/Slow EMA', color='green')
    
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend()
    plt.show()

    return nvda_adj_6mo1

# Set ticker and title
ticker = 'NVDA'
title_txt = "Triple Exponential Moving Average Crossover for NVDA stock"
label_txt = "NVDA Adj Close"

# Call the function to plot and get the DataFrame
nvda_adj_6mo1 = ewma3()

# Now nvda_adj_6mo1 contains the EMAs
print(nvda_adj_6mo1)


# In[283]:


# Define the function for buy/sell signals based on EMAs
def buy_sell_ewma3(data):
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False

    for i in range(0, len(data)):
        if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Adj Close'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short == True and data['Short'][i] > data['Middle'][i]:
            sell_list.append(data['Adj Close'][i])
            buy_list.append(np.nan)
            flag_short = False
        elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Adj Close'][i])
            sell_list.append(np.nan)
            flag_long = True
        elif flag_long == True and data['Short'][i] < data['Middle'][i]:
            sell_list.append(data['Adj Close'][i])
            buy_list.append(np.nan)
            flag_long = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    return buy_list, sell_list

# Calculate the EMAs and add them to the DataFrame
nvda_adj_6mo1 = ewma3()
nvda_adj_6mo1['Short'] = nvda_adj_6mo1['Adj Close'].ewm(span=5, adjust=False).mean()
nvda_adj_6mo1['Middle'] = nvda_adj_6mo1['Adj Close'].ewm(span=21, adjust=False).mean()
nvda_adj_6mo1['Long'] = nvda_adj_6mo1['Adj Close'].ewm(span=63, adjust=False).mean()

# Generate buy/sell signals
buy_signals, sell_signals = buy_sell_ewma3(nvda_adj_6mo1)

# Add buy/sell signals to the DataFrame
nvda_adj_6mo1['Buy_Signal'] = buy_signals
nvda_adj_6mo1['Sell_Signal'] = sell_signals

# Display the DataFrame with signals
print(nvda_adj_6mo1)


# In[284]:


# Define the function for buy/sell signals based on EMAs
def buy_sell_ewma3(data):
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False

    for i in range(0, len(data)):
        if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Adj Close'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short == True and data['Short'][i] > data['Middle'][i]:
            sell_list.append(data['Adj Close'][i])
            buy_list.append(np.nan)
            flag_short = False
        elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Adj Close'][i])
            sell_list.append(np.nan)
            flag_long = True
        elif flag_long == True and data['Short'][i] < data['Middle'][i]:
            sell_list.append(data['Adj Close'][i])
            buy_list.append(np.nan)
            flag_long = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    return buy_list, sell_list

# Extract data for the 6-month period
nvda_adj_6mo = nvda_sma[['Adj Close']]['2024-05-01':'2024-10-31']

# Calculate EMAs
nvda_adj_6mo['Short'] = nvda_adj_6mo['Adj Close'].ewm(span=5, adjust=False).mean()
nvda_adj_6mo['Middle'] = nvda_adj_6mo['Adj Close'].ewm(span=21, adjust=False).mean()
nvda_adj_6mo['Long'] = nvda_adj_6mo['Adj Close'].ewm(span=63, adjust=False).mean()

# Generate buy/sell signals
nvda_adj_6mo['Buy'] = buy_sell_ewma3(nvda_adj_6mo)[0]
nvda_adj_6mo['Sell'] = buy_sell_ewma3(nvda_adj_6mo)[1]

# Define the function for plotting buy/sell signals and EMAs
def buy_sell_ewma3_plot():
    sns.set(rc={'figure.figsize':(18, 10)})
    plt.plot(nvda_adj_6mo['Adj Close'], label=f"{label_txt}", color='blue', alpha=0.35)
    plt.plot(nvda_adj_6mo['Short'], label='Short/Fast EMA', color='red', alpha=0.35)
    plt.plot(nvda_adj_6mo['Middle'], label='Middle/Medium EMA', color='orange', alpha=0.35)
    plt.plot(nvda_adj_6mo['Long'], label='Long/Slow EMA', color='green', alpha=0.35)
    plt.scatter(nvda_adj_6mo.index, nvda_adj_6mo['Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(nvda_adj_6mo.index, nvda_adj_6mo['Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend()
    plt.show()

# Set ticker and title
ticker = 'NVDA'
title_txt = "Trading signals for NVDA stock"
label_txt = "NVDA Adj Close"

# Call the function to plot
buy_sell_ewma3_plot()


# Exponential Smoothing
# Single Exponential Smoothing, also known as Simple Exponential Smoothing, is a time series forecasting method for univariate data without a trend or seasonality. It requires an alpha parameter, also called the smoothing factor or smoothing coefficient, to control the rate at which the influence of the observations at prior time steps decay exponentially.

# In[285]:


# Exponential smoothing function
def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

# Function to plot exponential smoothing
def plot_exponential_smoothing(series, alphas):
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label=f"Alpha {alpha}")
    plt.plot(series.values, "c", label=f"{label_txt}")
    plt.xlabel('Days', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.grid(True)
    plt.show()

# Set ticker and title for NVDA
ticker = 'NVDA'
title_txt = "Single Exponential Smoothing for NVDA stock using 0.05 and 0.3 as alpha values"
label_txt = "NVDA Adj Close"

# Assuming nvda_sma is your dataframe containing NVDA stock data
plot_exponential_smoothing(nvda_sma['Adj Close'].loc['2024-01-01':'2024-12-31'], [0.05, 0.3])


# The smaller the smoothing factor (coefficient), the smoother the time series will be. As the smoothing factor approaches 0, we approach the moving average model so the smoothing factor of 0.05 produces a smoother time series than 0.3. This indicates slow learning (past observations have a large influence on forecasts). A value close to 1 indicates fast learning (that is, only the most recent values influence the forecasts).
# 
# Double Exponential Smoothing (Holt’s Linear Trend Model) is an extension being a recursive use of Exponential Smoothing twice where beta is the trend smoothing factor, and takes values between 0 and 1. It explicitly adds support for trends.

# In[286]:


# Double Exponential Smoothing function
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):  # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result

# Function to plot Double Exponential Smoothing
def plot_double_exponential_smoothing(series, alphas, betas):
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label=f"Alpha {alpha}, Beta {beta}")
    plt.plot(series.values, label=f"{label_txt}")
    plt.xlabel('Days', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.grid(True)
    plt.show()

# Set ticker and title for NVDA
ticker = 'NVDA'
title_txt = "Double Exponential Smoothing for NVDA stock with different alpha and beta values"
label_txt = "NVDA Adj Close"

# Assuming nvda_sma is your dataframe containing NVDA stock data
plot_double_exponential_smoothing(nvda_sma['Adj Close'].loc['2024-01-01':'2024-12-31'], alphas=[0.9, 0.02], betas=[0.9, 0.02])
 


# The third main type is Triple Exponential Smoothing (Holt Winters Method) which is an extension of Exponential Smoothing that explicitly adds support for seasonality, or periodic fluctuations.

# Moving average convergence divergence (MACD)
# The MACD is a trend-following momentum indicator turning two trend-following indicators, moving averages, into a momentum oscillator by subtracting the longer moving average from the shorter one.
# 
# It is useful although lacking one prediction element - because it is unbounded it is not particularly useful for identifying overbought and oversold levels. Traders can look for signal line crossovers, neutral/centreline crossovers (otherwise known as the 50 level) and divergences from the price action to generate signals.
# 
# The default parameters are 26 EMA of prices, 12 EMA of prices and a 9-moving average of the difference between the first two.

# In[290]:


# Function to plot the adjusted close price for a 3-month period
def adj_3mo():
    sns.set(rc={'figure.figsize': (15, 9)})
    nvda_sma['Adj Close'].loc['2024-05-15':'2024-08-15'].plot(label=f"{label_txt}")
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend()
    plt.show()

# Set title and label for NVDA
title_txt = "NVDA Adjusted Close Price from 1 Aug - 31 Oct 2024"
label_txt = "NVDA Adj Close"

# Call the function to plot
adj_3mo()


# In[334]:


# Extracting the data for the specified period
nvda_adj_3mo = nvda_sma[['Adj Close']]['2024-05-15':'2024-08-15']

# Calculate EMAs and MACD
ShortEMA = nvda_adj_3mo['Adj Close'].ewm(span=12, adjust=False).mean()
LongEMA = nvda_adj_3mo['Adj Close'].ewm(span=26, adjust=False).mean()
MACD = ShortEMA - LongEMA
signal = MACD.ewm(span=9, adjust=False).mean()

# Define the MACD plotting function
def macd():
    plt.figure(figsize=(15, 9))
    plt.plot(nvda_adj_3mo.index, MACD, label=macd_label_txt, color='red')
    plt.plot(nvda_adj_3mo.index, signal, label=sig_label_txt, color='blue')
    plt.title(title_txt, color='black', fontsize=20)
    plt.xticks(rotation=45)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.legend(loc='upper left')
    plt.show()

# Set title and labels
title_txt = 'MACD and Signal line for NVDA stock from 15 may - 15 Aug 2024'
macd_label_txt = "NVDA MACD"
sig_label_txt = "Signal Line"

# Call the function to plot
macd()


# When the MACD line crosses above the signal line this indicates a good time to buy.

# In[297]:


# Create new columns for the MACD and Signal Line data

nvda_adj_3mo['MACD'] = MACD
nvda_adj_3mo['Signal Line'] = signal
nvda_adj_3mo


# In[303]:


# Extracting the data for the specified period for NVDA
nvda_adj_3mo = nvda_sma[['Adj Close']]['2024-05-15':'2024-08-15']

# Calculate EMAs and MACD
ShortEMA = nvda_adj_3mo['Adj Close'].ewm(span=12, adjust=False).mean()
LongEMA = nvda_adj_3mo['Adj Close'].ewm(span=26, adjust=False).mean()
MACD = ShortEMA - LongEMA
signal = MACD.ewm(span=9, adjust=False).mean()

# Create a DataFrame with MACD and Signal Line
macd_signal_df = pd.DataFrame({
    'Adj Close': nvda_adj_3mo['Adj Close'],
    'MACD': MACD,
    'Signal Line': signal
})

# Function to signal when to buy and sell
def buy_sell_macd(df):
    Buy = []
    Sell = []
    flag = -1

    for i in range(len(df)):
        if df['MACD'][i] > df['Signal Line'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(df['Adj Close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif df['MACD'][i] < df['Signal Line'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(df['Adj Close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return (Buy, Sell)

# Create buy and sell columns
a = buy_sell_macd(macd_signal_df)
nvda_adj_3mo['Buy_Signal_Price'] = a[0]
nvda_adj_3mo['Sell_Signal_Price'] = a[1]

# Set labels and titles
ticker = 'NVDA'
title_txt = 'MACD and Signal line for NVDA stock from 15 May - 15 Aug 2024'
macd_label_txt = "NVDA MACD"
sig_label_txt = "Signal Line"

# Function to plot MACD and Signal Line
def macd():
    plt.figure(figsize=(15, 9))
    plt.plot(nvda_adj_3mo.index, MACD, label=f"{macd_label_txt}", color='red')
    plt.plot(nvda_adj_3mo.index, signal, label=f"{sig_label_txt}", color='blue')
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xticks(rotation=45)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.legend(loc='upper left')
    plt.show()

# Call the functions to plot and signal
macd()


# In[304]:


# Extract data for NVDA from May 15 to August 15, 2024
nvda_adj_3mo = nvda_sma[['Adj Close']]['2024-05-15':'2024-08-15']

# Calculate EMAs and MACD
ShortEMA = nvda_adj_3mo['Adj Close'].ewm(span=12, adjust=False).mean()
LongEMA = nvda_adj_3mo['Adj Close'].ewm(span=26, adjust=False).mean()
MACD = ShortEMA - LongEMA
signal = MACD.ewm(span=9, adjust=False).mean()

# Create new columns for the MACD and Signal Line data
nvda_adj_3mo['MACD'] = MACD
nvda_adj_3mo['Signal Line'] = signal

# Function to signal when to buy and sell
def buy_sell_macd(signal):
    Buy = []
    Sell = []
    flag = -1

    for i in range(len(signal)):
        if signal['MACD'][i] > signal['Signal Line'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal['Adj Close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal['MACD'][i] < signal['Signal Line'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal['Adj Close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return (Buy, Sell)

# Create buy and sell columns
a = buy_sell_macd(nvda_adj_3mo)
nvda_adj_3mo['Buy_Signal_Price'] = a[0]
nvda_adj_3mo['Sell_Signal_Price'] = a[1]

# Plot buy and sell signals
def buy_sell_macd_plot():
    plt.figure(figsize=(20, 10))
    plt.scatter(nvda_adj_3mo.index, nvda_adj_3mo['Buy_Signal_Price'], color='green', label='Buy', marker='^', alpha=1)
    plt.scatter(nvda_adj_3mo.index, nvda_adj_3mo['Sell_Signal_Price'], color='red', label='Sell', marker='v', alpha=1)
    plt.plot(nvda_adj_3mo['Adj Close'], label='Adj Close Price', alpha=0.35)
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Adj Close Price')
    plt.legend(loc='upper left')
    plt.show()

# Set labels and titles
ticker = 'NVDA'
title_txt = 'NVDA Adjusted Close Price Buy & Sell Signals'

# Call the function to plot
buy_sell_macd_plot()


# Momentum Strategies
# In momentum algorithmic trading strategies stocks have momentum (i.e. upward or downward trends) that we can detect and exploit.
# 
# Relative Strength Index (RSI)
# The RSI is a momentum indicator. A typical momentum strategy will buy stocks that have been showing an upward trend in hopes that the trend will continue, and make predictions based on whether the past recent values were going up or going down.
# 
# The RSI determines the level of overbought (70) and oversold (30) zones using a default lookback period of 14 i.e. it uses the last 14 values to calculate its values. The idea is to buy when the RSI touches the 30 barrier and sell when it touches the 70 barrier.

# In[308]:


# Extract data for NVDA from May 15 to August 15, 2024
nvda_adj_12mo = nvda_sma[['Adj Close']]['2024-01-01':'2024-12-31']

# Calculate the RSI
delta = nvda_adj_12mo['Adj Close'].diff(1)
up = delta.copy()
down = delta.copy()

up[up < 0] = 0
down[down > 0] = 0

period = 14

# Calculate average gain and average loss
AVG_Gain = up.rolling(window=period).mean()
AVG_Loss = down.abs().rolling(window=period).mean()

# Calculate RSI based on SMA
RS = AVG_Gain / AVG_Loss
RSI = 100.0 - (100.0 / (1.0 + RS))

# Create dataframe with Adjusted Close and RSI
new_df = pd.DataFrame()
new_df['Adj Close'] = nvda_adj_12mo['Adj Close']
new_df['RSI'] = RSI

# Function to plot Adjusted Close price
def adj_close_12mo():
    sns.set(rc={'figure.figsize':(20, 10)})
    plt.plot(new_df.index, new_df['Adj Close'], label='Adj Close')
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend(loc='upper left')
    plt.show()

# Function to plot RSI
def rsi():
    sns.set(rc={'figure.figsize':(20, 10)})
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('RSI', color='black', fontsize=15)
    RSI.plot()
    plt.show()

# Function to plot RSI with significant levels
def rsi_sma():
    plt.figure(figsize=(20, 10))
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.plot(new_df.index, new_df['RSI'], label='RSI')
    plt.axhline(0, linestyle='--', alpha=0.5, color='gray')
    plt.axhline(10, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(20, linestyle='--', alpha=0.5, color='green')
    plt.axhline(30, linestyle='--', alpha=0.5, color='red')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(80, linestyle='--', alpha=0.5, color='green')
    plt.axhline(90, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(100, linestyle='--', alpha=0.5, color='gray')
    plt.xlabel('Date', color='black', fontsize=15)
    plt.show()

# Set labels and titles
ticker = 'NVDA'
title_txt = 'NVDA Adjusted Close Price from 01 Jan - 31 Dec 2024'

# Call the functions to plot
adj_close_12mo()
rsi()
title_txt = 'NVDA RSI based on SMA'
rsi_sma()


# In[311]:


# Define period for RSI calculation
period = 14

# Update the data for NVDA for the period May 15 to August 15, 2024
nvda_adj_3mo = nvda_sma[['Adj Close']]['2024-01-01':'2024-12-31']

# Calculate the daily price changes
delta = nvda_adj_3mo['Adj Close'].diff(1)

# Get positive gains (up) and negative gains (down)
up = delta.copy()
down = delta.copy()

up[up < 0] = 0
down[down > 0] = 0 

# Calculate EWMA average gain and average loss
AVG_Gain2 = up.ewm(span=period).mean()
AVG_Loss2 = down.abs().ewm(span=period).mean()

# Calculate RSI based on EWMA
RS2 = AVG_Gain2 / AVG_Loss2
RSI2 = 100.0 - (100.0 / (1.0 + RS2))

# Create DataFrame for Adjusted Close and EWMA RSI
new_df2 = pd.DataFrame()
new_df2['Adj Close'] = nvda_adj_3mo['Adj Close']
new_df2['RSI2'] = RSI2

# Function to plot RSI with significant levels
def rsi_ewma():
    plt.figure(figsize=(20, 10))
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('RSI', color='black', fontsize=15)
    plt.plot(new_df2.index, new_df2['RSI2'], label='RSI2')
    plt.axhline(0, linestyle='--', alpha=0.5, color='gray')
    plt.axhline(10, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(20, linestyle='--', alpha=0.5, color='green')
    plt.axhline(30, linestyle='--', alpha=0.5, color='red')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(80, linestyle='--', alpha=0.5, color='green')
    plt.axhline(90, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(100, linestyle='--', alpha=0.5, color='gray')
    plt.legend(loc='upper left')
    plt.show()

# Set title for the plot
title_txt = 'NVDA RSI based on EWMA from Jan 01 - dec 31, 2024'

# Call the function to plot
rsi_ewma()

# It appears that RSI value dips below the 20 significant level in January 2024 indicating that the stock was oversold and presented a buying opportunity for an investor before a price rise.

# Money Flow Index (MFI)
# Money Flow Index (MFI) is a technical oscillator, and momentum indicator, that uses price and volume data for identifying overbought or oversold signals in an asset. It can also be used to spot divergences which warn of a trend change in price. The oscillator moves between 0 and 100 and a reading of above 80 implies overbought conditions, and below 20 implies oversold conditions.
# 
# It is related to the Relative Strength Index (RSI) but incorporates volume, whereas the RSI only considers price.

# Define period for MFI calculation
period = 14

# Extract data for NVDA for the period May 15 to August 15, 2024
nvda_3mo = nvda_sma[['Close', 'High', 'Low', 'Volume']]['2024-01-01':'2024-12-31']

# Function to plot Close Price
def nvda_close_plot():
    plt.figure(figsize=(20, 10))
    plt.plot(nvda_3mo['Close'])
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Close Price', color='black', fontsize=15)
    plt.legend([label_txt], loc='upper left')
    plt.show()

# Calculate typical price
typical_price = (nvda_3mo['Close'] + nvda_3mo['High'] + nvda_3mo['Low']) / 3

# Calculate the money flow
money_flow = typical_price * nvda_3mo['Volume']

# Get all positive and negative money flows
positive_flow = []
negative_flow = []

# Loop through typical price
for i in range(1, len(typical_price)):
    if typical_price[i] > typical_price[i-1]:
        positive_flow.append(money_flow[i-1])
        negative_flow.append(0)
    elif typical_price[i] < typical_price[i-1]:
        negative_flow.append(money_flow[i-1])
        positive_flow.append(0)
    else:
        positive_flow.append(0)
        negative_flow.append(0)

# Get all positive and negative money flows within the same time period
positive_mf = []
negative_mf = []

for i in range(period-1, len(positive_flow)):
    positive_mf.append(sum(positive_flow[i + 1 - period : i+1]))
for i in range(period-1, len(negative_flow)):
    negative_mf.append(sum(negative_flow[i + 1 - period : i+1]))

# Calculate Money Flow Index (MFI)
mfi = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))

# Create DataFrame for MFI
df2 = pd.DataFrame()
df2['MFI'] = mfi

# Function to plot MFI
def mfi_plot():
    plt.figure(figsize=(20, 10))
    plt.plot(df2['MFI'], label='MFI')
    plt.axhline(10, linestyle='--', color='orange')
    plt.axhline(20, linestyle='--', color='blue')
    plt.axhline(80, linestyle='--', color='blue')
    plt.axhline(90, linestyle='--', color='orange')
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Time periods', color='black', fontsize=15)
    plt.ylabel('MFI Values', color='black', fontsize=15)
    plt.legend(loc='upper left')
    plt.show()

# Create new DataFrame with MFI
new_mfi_df = pd.DataFrame()
new_mfi_df = nvda_3mo[period:]
new_mfi_df['MFI'] = mfi

# Function to get buy and sell signals
def get_signal(data, high, low):
    buy_signal = []
    sell_signal = []

    for i in range(len(data['MFI'])):
        if data['MFI'][i] > high:
            buy_signal.append(np.nan)
            sell_signal.append(data['Close'][i])
        elif data['MFI'][i] < low:
            buy_signal.append(data['Close'][i])
            sell_signal.append(np.nan)
        else:
            sell_signal.append(np.nan)
            buy_signal.append(np.nan)

    return (buy_signal, sell_signal)

# Add new columns (Buy & Sell)
new_mfi_df['Buy'] = get_signal(new_mfi_df, 80, 20)[0]
new_mfi_df['Sell'] = get_signal(new_mfi_df, 80, 20)[1]

# Function to plot buy and sell signals
def mfi_buy_sell_plot():
    plt.figure(figsize=(20, 10))
    plt.plot(new_mfi_df['Close'], label='Close Price', alpha=0.5)
    plt.scatter(new_mfi_df.index, new_mfi_df['Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(new_mfi_df.index, new_mfi_df['Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Close Price', color='black', fontsize=15)
    plt.legend(loc='upper left')
    plt.show()

# Set title for the plots
title_txt = "NVDA MFI and Trading Signals from Jun 01 - Dec 31, 2024"
label_txt = "NVDA Close Price"

# Call functions to plot
nvda_close_plot()
mfi_plot()
mfi_buy_sell_plot()


#Stochastic Oscillator
# The stochastic oscillator is a momentum indicator comparing the closing price of a security to the range of its prices over a certain period of time and is one of the best-known momentum indicators along with RSI and MACD.
# 
# The intuition is that in a market trending upward, prices will close near the high, and in a market trending downward, prices close near the low.
# 
# The stochastic oscillator is plotted within a range of zero and 100. The default parameters are an overbought zone of 80, an oversold zone of 20 and well-used lookbacks period of 14 and 5 which can be used simultaneously. The oscillator has two lines, the %K and %D, where the former measures momentum and the latter measures the moving average of the former. The %D line is more important of the two indicators and tends to produce better trading signals which are created when the %K crosses through the %D.

# In[315]:


# Define period for the rolling windows
period = 14

# Assuming `nvda` is the DataFrame with NVDA stock data for the period May 15 to August 15, 2024
nvda_so = nvda_sma.copy()
nvda_so = nvda_so['2024-01-01':'2024-12-31']

# Create the "L14" column in the DataFrame
nvda_so['L14'] = nvda_so['Low'].rolling(window=period).min()

# Create the "H14" column in the DataFrame
nvda_so['H14'] = nvda_so['High'].rolling(window=period).max()

# Create the "%K" column in the DataFrame
nvda_so['%K'] = 100 * ((nvda_so['Close'] - nvda_so['L14']) / (nvda_so['H14'] - nvda_so['L14']))

# Create the "%D" column in the DataFrame
nvda_so['%D'] = nvda_so['%K'].rolling(window=3).mean()

# Plot Close price and Stochastic Oscillator
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5)

nvda_so['Close'].plot(ax=axes[0])
axes[0].set_title('Close Price')
axes[0].set_ylabel('Close Price')

nvda_so[['%K', '%D']].plot(ax=axes[1])
axes[1].set_title('Stochastic Oscillator')
axes[1].set_ylabel('Oscillator Value')
plt.show()

# Create a column for sell entry signal
nvda_so['Sell Entry'] = ((nvda_so['%K'] < nvda_so['%D']) & (nvda_so['%K'].shift(1) > nvda_so['%D'].shift(1))) & (nvda_so['%D'] > 80)

# Create a column for sell exit signal
nvda_so['Sell Exit'] = ((nvda_so['%K'] > nvda_so['%D']) & (nvda_so['%K'].shift(1) < nvda_so['%D'].shift(1)))

# Create a column for buy entry signal
nvda_so['Buy Entry'] = ((nvda_so['%K'] > nvda_so['%D']) & (nvda_so['%K'].shift(1) < nvda_so['%D'].shift(1))) & (nvda_so['%D'] < 20)

# Create a column for buy exit signal
nvda_so['Buy Exit'] = ((nvda_so['%K'] < nvda_so['%D']) & (nvda_so['%K'].shift(1) > nvda_so['%D'].shift(1)))

# Create a placeholder column for short positions
nvda_so['Short'] = np.nan
nvda_so.loc[nvda_so['Sell Entry'], 'Short'] = -1
nvda_so.loc[nvda_so['Sell Exit'], 'Short'] = 0

# Set initial position to flat
nvda_so['Short'].iloc[0] = 0

# Forward fill the position column
nvda_so['Short'] = nvda_so['Short'].fillna(method='ffill')

# Create a placeholder column for long positions
nvda_so['Long'] = np.nan
nvda_so.loc[nvda_so['Buy Entry'], 'Long'] = 1
nvda_so.loc[nvda_so['Buy Exit'], 'Long'] = 0

# Set initial position to flat
nvda_so['Long'].iloc[0] = 0

# Forward fill the position column
nvda_so['Long'] = nvda_so['Long'].fillna(method='ffill')

# Add Long and Short positions together to get final strategy position
nvda_so['Position'] = nvda_so['Long'] + nvda_so['Short']

# Plot the position through time
nvda_so['Position'].plot(figsize=(20, 10))
plt.title('Strategy Position')
plt.ylabel('Position')
plt.show()

# Set up a column holding the daily NVDA returns
nvda_so['Market Returns'] = nvda_so['Close'].pct_change()

# Create column for Strategy Returns
nvda_so['Strategy Returns'] = nvda_so['Market Returns'] * nvda_so['Position'].shift(1)

# Plot strategy returns versus NVDA returns
nvda_so[['Strategy Returns', 'Market Returns']].cumsum().plot(figsize=(20, 10))
plt.title('Strategy Returns versus NVDA Returns')
plt.ylabel('Cumulative Returns')
plt.show()


#  Rate of Change (ROC)  ---------------          Candlestick, ROC and Volume plot        ----------------------------------
# The ROC indicator is a pure momentum oscillator. The ROC calculation compares the current price with the price "n" periods ago e.g. when we compute the ROC of the daily price with a 9-day lag, we are simply looking at how much, in percentage, the price has gone up (or down) compared to 9 days ago. Like other momentum indicators, ROC has overbought and oversold zones that may be adjusted according to market conditions.

# In[325]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
# Assuming `nvda_sma` is the DataFrame with NVDA stock data
nvda_smanvda_roc = nvda_sma.copy()
nvda_roc_12mo = nvda_roc['2024-01-01':'2024-12-31']

# Calculate ROC
nvda_roc_12mo['ROC'] = (nvda_roc_12mo['Adj Close'] / nvda_roc_12mo['Adj Close'].shift(9) - 1) * 100

# Select data for the last 100 days of 2024
nvda_roc_100d = nvda_roc_12mo[-100:]
dates = nvda_roc_100d.index
price = nvda_roc_100d['Adj Close']
roc = nvda_roc_100d['ROC']

# Plot Price and ROC
fig, (price_ax, roc_ax) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.subplots_adjust(hspace=0)

plt.rcParams.update({'font.size': 14})

# Price subplot
price_ax.plot(dates, price, color='blue', linewidth=2, label="Adj Closing Price")
price_ax.legend(loc="upper left", fontsize=12)
price_ax.set_ylabel("Price")
price_ax.set_title("NVDA Daily Price", fontsize=24)
price_ax.set_facecolor((.94, .95, .98))

# ROC subplot
roc_ax.plot(dates, roc, color='k', linewidth=1, alpha=0.7, label="9-Day ROC")
roc_ax.legend(loc="upper left", fontsize=12)
roc_ax.set_ylabel("% ROC")
roc_ax.set_facecolor((.98, .97, .93))

# Adding a horizontal line at the zero level in the ROC subplot
roc_ax.axhline(0, color=(.5, .5, .5), linestyle='--', alpha=0.5)

# Filling the areas between the indicator and the zero line
roc_ax.fill_between(dates, 0, roc, where=(roc >= 0), color='g', alpha=0.3, interpolate=True)
roc_ax.fill_between(dates, 0, roc, where=(roc < 0), color='r', alpha=0.3, interpolate=True)

# Formatting the date labels and ROC y-axis
roc_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
roc_ax.yaxis.set_major_formatter(mticker.PercentFormatter())

# Adding a grid to both subplots
price_ax.grid(True, linestyle='--', alpha=0.5)
roc_ax.grid(True, linestyle='--', alpha=0.5)

# Adding margins around the plots
price_ax.margins(0.05, 0.2)
roc_ax.margins(0.05, 0.2)

# Hiding tick marks from the horizontal and vertical axis
price_ax.tick_params(left=False, bottom=False)
roc_ax.tick_params(left=False, bottom=False, labelrotation=45)

# Hiding all the spines for the price subplot
for s in price_ax.spines.values():
    s.set_visible(False)

# Hiding all the spines for the ROC subplot
for s in roc_ax.spines.values():
    s.set_visible(False)

# Reinstate a spine in between the two subplots
roc_ax.spines['top'].set_visible(True)
roc_ax.spines['top'].set_linewidth(1.5)

# Candlestick and volume plot
mpf.plot(nvda_roc_100d, type='candle', style='yahoo', figsize=(15, 8), title="NVDA Daily Price", volume=True)

# Combined Candlestick and ROC plot
roc_plot = mpf.make_addplot(roc, panel=2, ylabel='ROC')
mpf.plot(nvda_roc_100d, type='candle', style='yahoo', figsize=(15, 8), addplot=roc_plot, title="NVDA Daily Price", volume=True)


# Volatility trading strategies
# Volatility trading involves predicting the stability of an asset’s value. Instead of trading on the price rising or falling, traders take a position on whether it will move in any direction.

# Bollinger Bands
# A Bollinger Band is a volatility indicator based on based on the correlation between the normal distribution and stock price and can be used to draw support and resistance curves. It is defined by a set of lines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of the security's price, but can be adjusted to user preferences.
# 
# By default it calculates a 20-period SMA (the middle band), an upper band two standard deviations above the the moving average and a lower band two standard deviations below it.
# 
# If the price moves above the upper band this could indicate a good time to sell, and if it moves below the lower band it could be a good time to buy.
# 
# Whereas the RSI can only be used as a confirming factor inside a ranging market, not a trending market, by using Bollinger bands we can calculate the widening variable, or moving spread between the upper and the lower bands, that tells us if prices are about to trend and whether the RSI signals might not be that reliable.
# 
# Despite 90% of the price action happening between the bands, however, a breakout is not necessarily a trading signal as it provides no clue as to the direction and extent of future price movement.

# In[328]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming `nvda_12mo` is the DataFrame with NVDA stock data
nvda_12mo_bb = nvda_roc_12mo.copy()

# Parameters
period = 20

# Calculate Bollinger Bands
nvda_12mo_bb['SMA'] = nvda_12mo_bb['Close'].rolling(window=period).mean()
nvda_12mo_bb['STD'] = nvda_12mo_bb['Close'].rolling(window=period).std()
nvda_12mo_bb['Upper'] = nvda_12mo_bb['SMA'] + (nvda_12mo_bb['STD'] * 2)
nvda_12mo_bb['Lower'] = nvda_12mo_bb['SMA'] - (nvda_12mo_bb['STD'] * 2)

# Keep relevant columns
column_list = ['Close', 'SMA', 'Upper', 'Lower']

# Plot Bollinger Bands
def bb_12mo():
    nvda_12mo_bb[column_list].plot(figsize=(20,10))
    plt.style.use('seaborn')
    plt.title('Bollinger Band for NVDA', color='black', fontsize=20)
    plt.ylabel('Close Price', color='black', fontsize=15)
    plt.show()

bb_12mo()

# Plot Bollinger Bands with shading
def bb_shaded():
    fig, ax = plt.subplots(figsize=(20, 10))
    x_axis = nvda_12mo_bb.index
    ax.fill_between(x_axis, nvda_12mo_bb['Upper'], nvda_12mo_bb['Lower'], color='grey')
    ax.plot(x_axis, nvda_12mo_bb['Close'], color='gold', lw=3, label='Close Price')
    ax.plot(x_axis, nvda_12mo_bb['SMA'], color='blue', lw=3, label='Simple Moving Average')
    ax.set_title('Bollinger Band For NVDA', color='black', fontsize=20)
    ax.set_xlabel('Date', color='black', fontsize=15)
    ax.set_ylabel('Close Price', color='black', fontsize=15)
    plt.xticks(rotation=45)
    ax.legend()
    plt.show()

bb_shaded()

# Prepare new DataFrame for signals
new_nvda_12mo_bb = nvda_12mo_bb[period-1:]

# Function to get buy and sell signals
def get_signal_bb(data):
    buy_signal = []
    sell_signal = []

    for i in range(len(data['Close'])):
        if data['Close'][i] > data['Upper'][i]:
            buy_signal.append(np.nan)
            sell_signal.append(data['Close'][i])
        elif data['Close'][i] < data['Lower'][i]:
            sell_signal.append(np.nan)
            buy_signal.append(data['Close'][i])
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
    
    return buy_signal, sell_signal

# Add buy and sell signals to DataFrame
new_nvda_12mo_bb['Buy'] = get_signal_bb(new_nvda_12mo_bb)[0]
new_nvda_12mo_bb['Sell'] = get_signal_bb(new_nvda_12mo_bb)[1]

# Plot all data with signals
def bb_alldata():
    fig, ax = plt.subplots(figsize=(20,10))
    x_axis = new_nvda_12mo_bb.index
    ax.fill_between(x_axis, new_nvda_12mo_bb['Upper'], new_nvda_12mo_bb['Lower'], color='grey')
    ax.plot(x_axis, new_nvda_12mo_bb['Close'], color='gold', lw=3, label='Close Price', alpha=0.5)
    ax.plot(x_axis, new_nvda_12mo_bb['SMA'], color='blue', lw=3, label='Moving Average', alpha=0.5)
    ax.scatter(x_axis, new_nvda_12mo_bb['Buy'], color='green', lw=3, label='Buy', marker='^', alpha=1)
    ax.scatter(x_axis, new_nvda_12mo_bb['Sell'], color='red', lw=3, label='Sell', marker='v', alpha=1)
    ax.set_title('Bollinger Band, Close Price, MA and Trading Signals for NVDA', color='black', fontsize=20)
    ax.set_xlabel('Date', color='black', fontsize=15)
    ax.set_ylabel('Close Price', color='black', fontsize=15)
    plt.xticks(rotation=45)
    ax.legend()
    plt.show()

bb_alldata()


# The Bollinger Bands technical indicator is an example of a mean reversion strategy.
# 
# Mean reversion strategies
# In mean reversion algorithmic trading strategies stocks return to their mean and we can exploit when it deviates from that mean.
# 
# These strategies usually involve selling into up moves and buying into down moves, a contrarian approach which assumes that the market has become oversold/overbought and prices will revert to their historical trends. This is almost the opposite of trend following where we enter in the direction of the strength and momentum, and momentum strategies such as buying stocks that have been showing an upward trend in hopes that the trend will continue, a continuation approach.

# Conclusion
# It is almost certainly better to choose technical indicators that complement each other, not just those that move in unison and generate the same signals. The intuition here is that the more indicators you have that confirm each other, the better your chances are to profit. This can be done by combining strategies to form a system, and looking for multiple signals.

# In[ ]:




