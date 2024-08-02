# app.py

from utils import prepare_data, analyze_data, plot_pairplot_and_heatmap
from utils import train_and_evaluate_model, train_and_evaluate_lstm_model, train_and_evaluate_dnn_model

# Tanımlamalar
tickers = ['JPM', 'BAC', 'WFC', 'C']
company_names = ['JPMORGAN', 'BANK OF AMERICA', 'Wells Fargo', 'Citigroup']

# Şirketlerin hisse senedi verilerini indirme ve işleme
dataframes = [prepare_data(ticker) for ticker in tickers]

# Tüm şirketler için analiz
for df, company_name in zip(dataframes, company_names):
    analyze_data(df, company_name)

# Günlük getirilerin histogramları
plt.figure(figsize=(12, 9))
for i, (df, company_name) in enumerate(zip(dataframes, company_names), 1):
    plt.subplot(2, 2, i)
    sns.histplot(df['Daily_Return'].dropna(), bins=50, kde=True, label=company_name, alpha=0.7)
    plt.title(f"{company_name} - Daily Returns")
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Verilerin birleştirilmesi
closing_df = pd.concat([df['Adj Close'] for df in dataframes], axis=1)
closing_df.columns = company_names
volume_df = pd.concat([df['Volume'] for df in dataframes], axis=1)
volume_df.columns = company_names

# Günlük getirilerin hesaplanması
daily_returns_df = closing_df.pct_change().dropna()

# Pairplot ve ısı haritaları
plot_pairplot_and_heatmap(daily_returns_df, "Daily Returns")
plot_pairplot_and_heatmap(closing_df, "Closing Prices")
plot_pairplot_and_heatmap(volume_df, "Volumes")

# Model eğitimi ve değerlendirme
for ticker, company_name in zip(tickers, company_names):
    train_and_evaluate_model(ticker, company_name)
    train_and_evaluate_lstm_model(ticker, company_name)
    train_and_evaluate_dnn_model(ticker, company_name)