# utils.py

# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# yfinance kütüphanesini pandas data reader üzerinde kullanma
yf.pdr_override()

# Grafik stil ayarları
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Tarih aralığı
end = datetime.now()
start = datetime(end.year - 7, end.month, end.day)

# Veri hazırlama fonksiyonu
def prepare_data(ticker):
    stock = pdr.get_data_yahoo(ticker, start=start, end=end)
    
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

# Veri analiz fonksiyonu
def analyze_data(df, company_name):
    # Kapanış fiyatlarının yıllık grafiği
    df['Adj Close'].plot(figsize=(10, 6), title=f"{company_name} - Adjusted Close Price", legend=True)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.show()

    # Hacim grafiği
    df['Volume'].plot(figsize=(10, 6), title=f"{company_name} - Volume", legend=True)
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.show()

    # Hareketli ortalamaların grafiği
    df[['Adj Close', '10_day_MA', '20_day_MA', '50_day_MA']].plot(figsize=(10, 6), title=f"{company_name} - Moving Averages")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Pairplot ve ısı haritaları
def plot_pairplot_and_heatmap(df, title):
    sns.pairplot(df, kind='reg')
    plt.suptitle(f'Pairplot of {title}', y=1.02)
    plt.show()
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='summer')
    plt.title(f'Correlation of {title}')
    plt.show()

# Model eğitimi ve değerlendirme fonksiyonu
def train_and_evaluate_model(ticker, company_name):
    df = prepare_data(ticker)
    X = df[['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']]
    y = df['Adj Close']

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
    print(f"{company_name} ({ticker}):")
    print(f"Random Forest - MSE: {mse_rf}, MAE: {mae_rf}")
    print(f"Gradient Boosting - MSE: {mse_gbr}, MAE: {mae_gbr}\n")

    # Tahmin grafikleri
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', alpha=0.7)
    plt.plot(df.index[-len(y_test):], y_pred_rf, label='Predicted Prices (RF)', alpha=0.7)
    plt.plot(df.index[-len(y_test):], y_pred_gbr, label='Predicted Prices (GBR)', alpha=0.7)
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {company_name} ({ticker})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# LSTM modeli eğitimi ve değerlendirme
def train_and_evaluate_lstm_model(ticker, company_name):
    df = prepare_data(ticker)
    X = df[['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']]
    y = df['Adj Close']

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

    print(f"{company_name} ({ticker}) - LSTM Model:")
    print(f"MSE: {mse_lstm}, MAE: {mae_lstm}")

    # Tahmin grafikleri
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', alpha=0.7)
    plt.plot(df.index[-len(y_test):], y_pred_lstm, label='Predicted Prices (LSTM)', alpha=0.7)
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {company_name} ({ticker}) - LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# DNN modeli eğitimi ve değerlendirme
def train_and_evaluate_dnn_model(ticker, company_name):
    df = prepare_data(ticker)
    X = df[['10_day_MA', '20_day_MA', '50_day_MA', 'Daily_Return']]
    y = df['Adj Close']

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

    y_pred_dnn = dnn_model.predict(X_test_scaled).flatten()

    mse_dnn = mean_squared_error(y_test, y_pred_dnn)
    mae_dnn = mean_absolute_error(y_test, y_pred_dnn)

    print(f"{company_name} ({ticker}) - DNN Model:")
    print(f"MSE: {mse_dnn}, MAE: {mae_dnn}")

    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', alpha=0.7)
    plt.plot(df.index[-len(y_test):], y_pred_dnn, label='Predicted Prices (DNN)', alpha=0.7)
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {company_name} ({ticker}) - DNN')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()e']

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

    print(f"{company_name} ({ticker}) - LSTM Model:")
    print(f"MSE: {mse_lstm}, MAE: {mae_lstm}")

    # Tahmin grafikleri
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test):],