#!/usr/bin/env python
# coding: utf-8

# # Data Project - Stock Market Analysis
# # TESSA NEJLA AYVAZOGLU
# # Sprint1 is done.
# ## Objective
# 
# This notebook explores stock market data, focusing on technology giants like Apple, Amazon, Google, and Microsoft. It demonstrates the use of yfinance to retrieve stock information and visualization techniques with Seaborn and Matplotlib. The analysis includes assessing stock risk using historical performance data and predicting future prices using a Linear Regression model.
# 
# ![Stock Market Reactions to Election](image/unleashing-the-bulls-how-the-stock-market-achieved-unprecedented-record-levels2.jpg)

# # Task Breakdown
# - Identify reliable market data APIs
# - Develop scripts/tools for data ingestion
# - Clean and preprocess collected data
# - Standardize data formats
# - Explore data visualization techniques
# - Perform exploratory data analysis (EDA)
# - Extract relevant features from raw financial data
# - Implement data transformation techniques
# - Split the preprocessed data into training, validation, and test sets
# - Document data collection and preprocessing procedures

# # Step-by-Step Implementation

# ## 1. Identify reliable market data APIs

# ##### We'll use the yfinance library, which provides a Pythonic interface to Yahoo Finance, a reliable source for historical market data.

# In[18]:


# Install necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime


yf.pdr_override()

# Set plotting styles
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')



# # 2. Develop scripts/tools for data ingestion
# ##### We'll create a script to download stock data for Apple, Amazon, Google, and Microsoft.

# In[20]:


# Define the tickers and corresponding company names
tickers = ['JPM', 'BAC', 'WFC', 'C']
company_names = ['JPMORGAN', 'BANK OF AMERICA', 'Wells Fargo', 'Citigroup']

# Define the date range
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
# Initialize an empty list to hold the DataFrames
dataframes = []    


# # 3. Clean and preprocess collected data
# ##### Add company names and concatenate data into a single DataFrame.

# In[21]:


# Download stock data for each company and process it
for ticker, company in zip(tickers, company_names):
    # Download the data
    df = yf.download(ticker, start=start, end=end)
    df.ffill(inplace=True)  # Fill missing values using forward fill
    
    # Reset and set index to ensure consistent date format
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Add a column with the company name
    df['company_name'] = company
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
df_combined = pd.concat(dataframes)

# Shuffle the data and get a random sample of the last 10 rows
df_combined = df_combined.sample(frac=1).reset_index(drop=True)
last_10_rows = df_combined.tail(10)

# Print the last 10 rows with company names
print("Displaying the last 10 rows of the shuffled DataFrame:\n")
for index, row in last_10_rows.iterrows():
    print(f"Company: {row['company_name']}, Date: {row.name}, Open: {row['Open']:.2f}, High: {row['High']:.2f}, "
          f"Low: {row['Low']:.2f}, Close: {row['Close']:.2f}, Adj Close: {row['Adj Close']:.2f}, "
          f"Volume: {row['Volume']}")

# Plot stock prices for each company
plt.figure(figsize=(14, 7))
for company in company_names:
    plt.plot(df_combined[df_combined['company_name'] == company].index,
             df_combined[df_combined['company_name'] == company]['Close'], label=company)
plt.title('Stock Prices Over the Last Year')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# # 4. Standardize data formats
# ##### Ensure all columns have consistent formats and handle missing values.

# In[23]:


df = df.reset_index()
df = df.fillna(method='ffill')


# # 5. Explore data visualization techniques
# ##### Visualize the closing price and volume of sales.

# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
 
dataframes = [JPM, BAC, WFC, C]
 
# Plotting Adjusted Close Prices for each company
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=0.9, bottom=0.1)  # Adjust the subplot layout
for i, df in enumerate(dataframes, 1):
    plt.subplot(2, 2, i)
    df['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.title(f"Closing Price of {company_names[i - 1]}")
plt.tight_layout(pad=2.0)
plt.show()

# Plotting Volume for each company
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=0.9, bottom=0.1)  # Adjust the subplot layout
for i, df in enumerate(dataframes, 1):
    plt.subplot(2, 2, i)
    df['Volume'].plot()
    plt.ylabel('Volume')
    plt.title(f"Sales Volume for {company_names[i - 1]}")
plt.tight_layout(pad=2.0)
plt.show()


# # 6. Perform exploratory data analysis (EDA)
# ##### Calculate and plot moving averages.

# In[34]:


# Download data for each ticker
dataframes = [yf.download(ticker, start=start, end=end) for ticker in tickers]

# Calculate moving averages for each company
ma_day = [10, 20, 50]
for ma in ma_day:
    for df in dataframes:
        column_name = f"MA for {ma} days"
        df[column_name] = df['Adj Close'].rolling(ma).mean()

# Create subplots for the moving averages
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Define company plotting details
plot_details = zip(dataframes, company_names)

# Loop through each company data to plot
for ax, (df, company_name) in zip(axes.flatten(), plot_details):
    df[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=ax)
    ax.set_title(f'{company_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True)

# Adjust layout to prevent overlap
fig.tight_layout(pad=2.0)

# Show the plot
plt.show()


# # 7. Extract relevant features from raw financial data
# ##### Calculate daily returns and visualize them.

# In[40]:


# Define the date range
end = pd.Timestamp.now()
start = end - pd.DateOffset(years=1)

# Initialize an empty list to hold the DataFrames
company_list = []

# Download stock data for each company and process it
for ticker, company_name in zip(tickers, company_names):
    # Download the data
    df = yf.download(ticker, start=start, end=end)
    df.ffill(inplace=True)  # Fill missing values using forward fill
    
    # Reset and set index to ensure consistent date format
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Calculate daily returns
    df['Daily Return'] = df['Adj Close'].pct_change()
    
    # Add a column with the company name
    df['Company Name'] = company_name
    
    # Append the DataFrame to the list
    company_list.append(df)

# Plotting daily returns for each company
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

for ax, df, company_name in zip(axes.flatten(), company_list, company_names):
    df['Daily Return'].plot(ax=ax, legend=True, linestyle='--', marker='o', label='Daily Return')
    ax.set_title(company_name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Return')
    ax.grid(True)
    ax.legend()

fig.tight_layout()

# Plotting histograms for daily returns of each company
plt.figure(figsize=(12, 9))

for i, df in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    df['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.title(company_names[i - 1])

plt.tight_layout()

# Show plots
plt.show()


# In[70]:


# Define the tickers and corresponding company names
tickers = ['JPM', 'BAC', 'WFC', 'C']
company_names = ['JPMORGAN', 'BANK OF AMERICA', 'Wells Fargo', 'Citigroup']

# Define the date range
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download stock data for the past year for each company
company_list = []
for ticker, company_name in zip(tickers, company_names):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)  # Drop rows with missing values
    df['Company'] = company_name  # Add a column with the company name
    company_list.append(df)

# Calculate daily returns for each company
for company in company_list:
    company['Daily Return (%)'] = company['Adj Close'].pct_change() * 100

# Plot histograms for daily returns of each company
plt.figure(figsize=(12, 8))
for i, company in enumerate(company_list):
    sns.histplot(company['Daily Return (%)'].dropna(), bins=20, kde=True, label=company_names[i], alpha=0.7)

plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Returns for Selected Companies (Past Year)')
plt.legend()
plt.tight_layout()

# Initialize an empty list to hold dictionaries of performance summary
performance_summary = []

# Populate performance summary with average daily returns
for i, company in enumerate(company_list):
    avg_daily_return = company['Daily Return (%)'].mean()
    performance_summary.append({'Company': company_names[i], 'Average Daily Return (%)': avg_daily_return})

# Convert the list of dictionaries to a DataFrame
performance_summary_df = pd.DataFrame(performance_summary)

# Identify best and worst performers
best_performer = performance_summary_df.loc[performance_summary_df['Average Daily Return (%)'].idxmax()]
worst_performer = performance_summary_df.loc[performance_summary_df['Average Daily Return (%)'].idxmin()]

# Display performance summary
print("Performance Summary:")
print(performance_summary_df)

# Print best and worst performers
print("\nBest Performer:")
print(best_performer)
print("\nWorst Performer:")
print(worst_performer)

# Display additional information or analysis as needed

plt.show()


# # 8. Implement data transformation techniques
# ##### Calculate correlations and visualize them using heatmaps and pair plots.

# In[71]:


company_list = []
for ticker, company_name in zip(tickers, company_names):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)  # Drop rows with missing values
    df['Company'] = company_name  # Add a column with the company name
    company_list.append(df)

# Combine 'Adj Close' from each company into a single DataFrame
closing_df = pd.concat([company['Adj Close'] for company in company_list], axis=1)
closing_df.columns = company_names

# Combine 'Volume' from each company into a single DataFrame
volume_df = pd.concat([company['Volume'] for company in company_list], axis=1)
volume_df.columns = company_names

# Calculate daily returns
daily_returns_df = closing_df.pct_change().dropna()

# Display pairplot of daily returns
sns.pairplot(daily_returns_df, kind='reg')
plt.suptitle('Pairplot of Daily Returns', y=1.02)
plt.show()

# Display pairplot of closing prices
sns.pairplot(closing_df, kind='reg')
plt.suptitle('Pairplot of Closing Prices', y=1.02)
plt.show()

# Display pairplot of volumes
sns.pairplot(volume_df, kind='reg')
plt.suptitle('Pairplot of Volumes', y=1.02)
plt.show()

# Heatmap of daily returns correlation
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
sns.heatmap(daily_returns_df.corr(), annot=True, cmap='summer')
plt.title('Correlation of Daily Returns')

# Heatmap of closing price correlation
plt.subplot(3, 1, 2)
sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
plt.title('Correlation of Closing Prices')

# Heatmap of volume correlation
plt.subplot(3, 1, 3)
sns.heatmap(volume_df.corr(), annot=True, cmap='winter')
plt.title('Correlation of Volumes')

plt.tight_layout()
plt.show()

# Identify best and worst correlations
correlation_daily_returns = daily_returns_df.corr()
correlation_closing_prices = closing_df.corr()
correlation_volumes = volume_df.corr()

# Find the worst and best correlations
worst_corr_daily_returns = correlation_daily_returns.min().min()
best_corr_daily_returns = correlation_daily_returns.max().max()

worst_corr_closing_prices = correlation_closing_prices.min().min()
best_corr_closing_prices = correlation_closing_prices.max().max()

worst_corr_volumes = correlation_volumes.min().min()
best_corr_volumes = correlation_volumes.max().max()

# Print statements for correlation coefficients
print(f"Worst correlation of daily returns: {worst_corr_daily_returns}")
print(f"Best correlation of daily returns: {best_corr_daily_returns}")
print(f"Worst correlation of closing prices: {worst_corr_closing_prices}")
print(f"Best correlation of closing prices: {best_corr_closing_prices}")
print(f"Worst correlation of volumes: {worst_corr_volumes}")
print(f"Best correlation of volumes: {best_corr_volumes}")

# Define the correlation coefficients and thresholds
correlations = {
    'Best Daily Returns Correlation': best_corr_daily_returns,
    'Worst Daily Returns Correlation': worst_corr_daily_returns,
    'Best Closing Prices Correlation': best_corr_closing_prices,
    'Worst Closing Prices Correlation': worst_corr_closing_prices,
    'Best Volumes Correlation': best_corr_volumes,
    'Worst Volumes Correlation': worst_corr_volumes
}

# Define the labels and categories with thresholds
categories = {
    'Perfect Positive Correlation': [1.0],
    'Strong Positive Correlation': [0.8, 1.0],
    'Moderate Positive Correlation': [0.6, 0.8],
    'Weak Positive Correlation': [0.0, 0.6]
}

# Print categorization based on thresholds
print("\nCategorization based on thresholds:")
for label, value in correlations.items():
    categorized = False
    for cat_label, cat_range in categories.items():
        if len(cat_range) == 2 and cat_range[0] <= value <= cat_range[1]:
            print(f"{label}: {cat_label}")
            categorized = True
            break
        elif len(cat_range) == 1 and value == cat_range[0]:
            print(f"{label}: {cat_label}")
            categorized = True
            break
    if not categorized:
        print(f"{label}: Undefined category (check thresholds)")


# # 9. Split the preprocessed data into training, validation, and test sets
# ##### Split the data for model training and validation.

# In[74]:


# Calculate daily returns
tech_rets = closing_df.pct_change()

# Drop missing values from both tech_rets and closing_df
tech_rets_cleaned = tech_rets.dropna()
closing_df_cleaned = closing_df.dropna()

# Ensure both dataframes have the same number of rows
min_rows = min(tech_rets_cleaned.shape[0], closing_df_cleaned.shape[0])
tech_rets_cleaned = tech_rets_cleaned.iloc[:min_rows]
closing_df_cleaned = closing_df_cleaned.iloc[:min_rows]

# Create X (features) and y (targets)
X = tech_rets_cleaned.values
y = closing_df_cleaned.values

# Print the shapes of X and y for verification
print("Shape of X (features):", X.shape)
print("Shape of y (targets):", y.shape)

# Split the data into training (60%) and remaining (40%)
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, random_state=42)

# Split the remaining data into validation (50% of remaining, i.e., 20% of total) and test (50% of remaining, i.e., 20% of total)
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

# Print the shapes of the split data for verification
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)


# In[77]:


# Define colors for the histograms
train_colors = ['blue', 'green', 'orange', 'purple']
val_colors = ['navy', 'lime', 'darkorange', 'indigo']
test_colors = ['red', 'yellow', 'cyan', 'magenta']

# Plot the distribution of y_train
plt.figure(figsize=(12, 6))
for i, dataset in enumerate(y_train.T):
    stock_name = company_names[i]  # Use company_names for labeling
    plt.hist(dataset, bins=30, color=train_colors[i], alpha=0.7, label=stock_name)

plt.title('Distribution of y_train (Closing Prices)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot the distribution of y_val
plt.figure(figsize=(12, 6))
for i, dataset in enumerate(y_val.T):
    stock_name = company_names[i]
    plt.hist(dataset, bins=30, color=val_colors[i], alpha=0.7, label=stock_name)

plt.title('Distribution of y_val (Closing Prices)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot the distribution of y_test
plt.figure(figsize=(12, 6))
for i, dataset in enumerate(y_test.T):
    stock_name = company_names[i]
    plt.hist(dataset, bins=30, color=test_colors[i], alpha=0.7, label=stock_name)

plt.title('Distribution of y_test (Closing Prices)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# # 10. Document data collection and preprocessing procedures
# ##### Keep thorough documentation of each step for reproducibility.

# ## Documentation
# ### Data Collection
# - Data sourced from Yahoo Finance using yfinance library.
# - Stock symbols: AAPL, GOOG, MSFT, AMZN.
# - Time period: Last one year.
# 
# ### Data Preprocessing
# - Filled missing values using forward fill.
# - Added company name column.
# - Concatenated individual stock data into a single DataFrame.
# 
# ### Data Standardization
# - Ensured consistent date format.
# - Handled missing values.
# 
# ### Data Visualization
# - Plotted closing prices and volume of sales.
# - Calculated and plotted moving averages (10, 20, 50 days).
# - Visualized daily returns using histograms and line plots.
# 
# ### Feature Extraction
# - Calculated daily returns.
# - Analyzed correlations between stock returns using heatmaps and pair plots.
# 
# ### Data Splitting
# - Split data into training and test sets for model validation.

# ## Conclusion
# 
# In this notebook, we delved into the world of stock market data analysis. Here's a summary of what we explored:
# 
# - We learned how to retrieve stock market data from Yahoo Finance using the yfinance library.
# - Using Pandas, Matplotlib, and Seaborn, we visualized time-series data to gain insights into the stock market trends.
# - We measured the correlation between different stocks to understand how they move in relation to each other.
# - We assessed the risk associated with investing in a particular stock by analyzing its daily returns.
# - Lastly, we split the data into training and validation sets for further analysis and model training.
# 
# If you have any questions or need further clarification on any topic covered in this notebook, feel free to ask in the comments below. I'll be happy to assist you!
# 
# References:
# - [Investopedia on Correlation](https://www.investopedia.com/terms/c/correlation.asp)
# - file:///C:/Users/Admin/Desktop/C_AIML/semestert2/AI%20for%20Business/article1.pdf
# - https://medium.com/@ethan.duong1120/stock-data-analysis-project-python-1bf2c51b615f
#   

# In[13]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Reshape the data for LSTM input
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print("X_train:")
print(X_train[:10])  # Print the first 10 rows of X_train
print("\nX_test:")
print(X_test[:10])   # Print the first 10 rows of X_test
# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dense(units=1))

# Compile the model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=32)

# Make predictions
predictions_lstm = model_lstm.predict(X_test_lstm)


# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network model
model_nn = Sequential()
model_nn.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model_nn.add(Dense(units=32, activation='relu'))
model_nn.add(Dense(units=1))

# Compile the model
model_nn.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_nn.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions_nn = model_nn.predict(X_test)


# In[15]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_test:", np.unique(y_test))

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for i in range(y_train.shape[1]):
    plt.hist(y_train[:, i], bins=30, alpha=0.7, label=f'Stock {i+1}')

plt.title('Distribution of Target Variable (y_train)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Train a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# Make predictions
y_pred = regressor.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize predictions vs actual trend
# You can plot the predicted trend against the actual trend to visually compare
# Assuming you have a function to plot the trends similar to the one in your original code
# plot_trends(X_test, y_test, y_pred)


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Download stock data
yf.pdr_override()
tech_list = ['META', 'AAPL', 'NVDA', 'NFLX']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Function to prepare data for each stock
def prepare_data(ticker):
    stock = pdr.get_data_yahoo(ticker, start=start, end=end)
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change()
    stock.dropna(inplace=True)
    return stock

# Train and evaluate model for each stock
results = {}
for ticker in tech_list:
    print(f"Processing {ticker}...")
    stock = prepare_data(ticker)
    
    # Feature selection
    features = ['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']
    X = stock[features]
    y = stock['Adj Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training with hyperparameter tuning
    rf = RandomForestRegressor()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_rf.predict(X_test_scaled)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[ticker] = {'mse': mse, 'mae': mae, 'y_test': y_test, 'y_pred': y_pred}
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Display results
for ticker, result in results.items():
    print(f"{ticker}:")
    print(f"Mean Squared Error: {result['mse']}")
    print(f"Mean Absolute Error: {result['mae']}")
    print()


# In[17]:


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Define the start and end date for data download
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# List of tech stocks
tech_list = ['META', 'AAPL', 'NVDA', 'NFLX']

# Custom function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Custom function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

# Function to prepare data with additional features
def prepare_data(ticker):
    stock = yf.download(ticker, start=start, end=end)
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change()
    stock['RSI'] = calculate_rsi(stock['Adj Close'])
    stock['MACD'], stock['MACD_Signal'], stock['MACD_Hist'] = calculate_macd(stock['Adj Close'])
    stock.dropna(inplace=True)
    return stock

# Train and evaluate model for each stock
results = {}
for ticker in tech_list:
    print(f"Processing {ticker}...")
    stock = prepare_data(ticker)
    
    # Feature selection
    features = ['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
    X = stock[features]
    y = stock['Adj Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training with hyperparameter tuning
    gbr = GradientBoostingRegressor()
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_gbr = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_gbr.predict(X_test_scaled)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[ticker] = {'mse': mse, 'mae': mae, 'y_test': y_test, 'y_pred': y_pred}
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Display results
for ticker, result in results.items():
    print(f"{ticker}:")
    print(f"Mean Squared Error: {result['mse']}")
    print(f"Mean Absolute Error: {result['mae']}")
    print()


# In[18]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Define the start and end date for data download
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# List of tech stocks
tech_list = ['META', 'AAPL', 'NVDA', 'NFLX']

# Custom function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Custom function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

# Function to prepare data with additional features
def prepare_data(ticker):
    stock = yf.download(ticker, start=start, end=end)
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change()
    stock['RSI'] = calculate_rsi(stock['Adj Close'])
    stock['MACD'], stock['MACD_Signal'], stock['MACD_Hist'] = calculate_macd(stock['Adj Close'])
    
    # Adding Bollinger Bands
    stock['20_day_std'] = stock['Adj Close'].rolling(window=20).std()
    stock['Upper_Band'] = stock['20_day_MA'] + (stock['20_day_std'] * 2)
    stock['Lower_Band'] = stock['20_day_MA'] - (stock['20_day_std'] * 2)
    
    # Lag features
    stock['Lag_1'] = stock['Adj Close'].shift(1)
    stock['Lag_2'] = stock['Adj Close'].shift(2)
    
    stock.dropna(inplace=True)
    return stock

# Train and evaluate model for each stock
results = {}
for ticker in tech_list:
    print(f"Processing {ticker}...")
    stock = prepare_data(ticker)
    
    # Feature selection
    features = ['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                'Upper_Band', 'Lower_Band', 'Lag_1', 'Lag_2']
    X = stock[features]
    y = stock['Adj Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training with hyperparameter tuning
    gbr = GradientBoostingRegressor()
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_gbr = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_gbr.predict(X_test_scaled)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[ticker] = {'mse': mse, 'mae': mae, 'y_test': y_test, 'y_pred': y_pred}
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Display results
for ticker, result in results.items():
    print(f"{ticker}:")
    print(f"Mean Squared Error: {result['mse']}")
    print(f"Mean Absolute Error: {result['mae']}")
    print()


# In[19]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Define the start and end date for data download
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# List of tech stocks
tech_list = ['META', 'AAPL', 'NVDA', 'NFLX']

# Custom function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Custom function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

# Custom function to calculate On-Balance Volume (OBV)
def calculate_obv(data):
    obv = (np.sign(data['Adj Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

# Function to prepare data with additional features
def prepare_data(ticker):
    stock = yf.download(ticker, start=start, end=end)
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change()
    stock['RSI'] = calculate_rsi(stock['Adj Close'])
    stock['MACD'], stock['MACD_Signal'], stock['MACD_Hist'] = calculate_macd(stock['Adj Close'])
    stock['OBV'] = calculate_obv(stock)
    
    # Adding Bollinger Bands
    stock['20_day_std'] = stock['Adj Close'].rolling(window=20).std()
    stock['Upper_Band'] = stock['20_day_MA'] + (stock['20_day_std'] * 2)
    stock['Lower_Band'] = stock['20_day_MA'] - (stock['20_day_std'] * 2)
    
    # Lag features
    stock['Lag_1'] = stock['Adj Close'].shift(1)
    stock['Lag_2'] = stock['Adj Close'].shift(2)
    
    stock.dropna(inplace=True)
    return stock

# Train and evaluate model for each stock
results = {}
for ticker in tech_list:
    print(f"Processing {ticker}...")
    stock = prepare_data(ticker)
    
    # Feature selection
    features = ['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                'Upper_Band', 'Lower_Band', 'Lag_1', 'Lag_2', 'OBV']
    X = stock[features]
    y = stock['Adj Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training with hyperparameter tuning
    gbr = GradientBoostingRegressor()
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_gbr = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_gbr.predict(X_test_scaled)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[ticker] = {'mse': mse, 'mae': mae, 'y_test': y_test, 'y_pred': y_pred}
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Display results
for ticker, result in results.items():
    print(f"{ticker}:")
    print(f"Mean Squared Error: {result['mse']}")
    print(f"Mean Absolute Error: {result['mae']}")
    print()


# In[20]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Define the start and end date for data download
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# List of tech stocks
tech_list = ['META', 'AAPL', 'NVDA', 'NFLX']

# Custom function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Custom function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

# Custom function to calculate On-Balance Volume (OBV)
def calculate_obv(data):
    obv = (np.sign(data['Adj Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

# Function to prepare data with additional features
def prepare_data(ticker):
    stock = yf.download(ticker, start=start, end=end)
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change()
    stock['RSI'] = calculate_rsi(stock['Adj Close'])
    stock['MACD'], stock['MACD_Signal'], stock['MACD_Hist'] = calculate_macd(stock['Adj Close'])
    stock['OBV'] = calculate_obv(stock)
    
    # Adding Bollinger Bands
    stock['20_day_std'] = stock['Adj Close'].rolling(window=20).std()
    stock['Upper_Band'] = stock['20_day_MA'] + (stock['20_day_std'] * 2)
    stock['Lower_Band'] = stock['20_day_MA'] - (stock['20_day_std'] * 2)
    
    # Lag features
    stock['Lag_1'] = stock['Adj Close'].shift(1)
    stock['Lag_2'] = stock['Adj Close'].shift(2)
    
    stock.dropna(inplace=True)
    return stock

# Train and evaluate model for each stock
results = {}
for ticker in tech_list:
    print(f"Processing {ticker}...")
    stock = prepare_data(ticker)
    
    # Feature selection
    features = ['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                'Upper_Band', 'Lower_Band', 'Lag_1', 'Lag_2', 'OBV']
    X = stock[features]
    y = stock['Adj Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training with hyperparameter tuning
    gbr = GradientBoostingRegressor()
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_gbr = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_gbr.predict(X_test_scaled)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[ticker] = {'mse': mse, 'mae': mae, 'y_test': y_test, 'y_pred': y_pred}
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Display results
for ticker, result in results.items():
    print(f"{ticker}:")
    print(f"Mean Squared Error: {result['mse']}")
    print(f"Mean Absolute Error: {result['mae']}")
    print()


# In[21]:


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# Define the start and end date for data download
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Define the list of tech stocks
tech_list = ['META', 'AAPL', 'NVDA', 'NFLX']

# Function to prepare data
def prepare_data(ticker):
    stock = yf.download(ticker, start=start, end=end)
    stock['10_day_MA'] = stock['Adj Close'].rolling(window=10).mean()
    stock['20_day_MA'] = stock['Adj Close'].rolling(window=20).mean()
    stock['50_day_MA'] = stock['Adj Close'].rolling(window=50).mean()
    stock['Daily_Return'] = stock['Adj Close'].pct_change()
    stock.dropna(inplace=True)
    X = stock[['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']]
    y = stock['Adj Close']
    return X, y

# Train and evaluate model for each stock
best_models = {}
results = {}
for ticker in tech_list:
    print(f"Processing {ticker}...")
    X, y = prepare_data(ticker)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the model
    gbr = GradientBoostingRegressor()
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    best_models[ticker] = best_model
    
    # Save the best model
    joblib.dump(best_model, f"{ticker}_model.pkl")
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{ticker}:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print()

    # Store results
    results[ticker] = {"y_test": y_test, "y_pred": y_pred}

# Select the best model based on MSE
best_ticker, best_model = min(best_models.items(), key=lambda x: mean_squared_error(x[1].predict(X_test_scaled), y_test))

print(f"The best model is for {best_ticker} with MSE: {mean_squared_error(best_model.predict(X_test_scaled), y_test)}")

# Plot actual vs predicted for all stocks
plt.figure(figsize=(12, 8))
actual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
predicted_colors = ['pink', 'purple', 'yellow', 'turquoise']
for i, (ticker, result) in enumerate(results.items()):
    actual_color = actual_colors[i] if i < len(actual_colors) else actual_colors[i % len(actual_colors)]  # Cycle through the color list
    predicted_color = predicted_colors[i] if i < len(predicted_colors) else predicted_colors[i % len(predicted_colors)]  # Cycle through the color list
    plt.plot(result['y_test'].values, label=f'Actual Prices ({ticker})', color=actual_color)
    plt.plot(result['y_pred'], label=f'Predicted Prices ({ticker})', linestyle='--', color=predicted_color)
plt.legend()
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

