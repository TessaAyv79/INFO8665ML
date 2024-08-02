import sqlite3
from datetime import datetime
import yfinance as yf

# SQLite database connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Function to create tables if not exists
def create_tables():
    conn = get_db_connection()  # Call to the function get_db_connection
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            name TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            user_id INTEGER,
            selection_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER,
            recommendation TEXT,
            analysis_date TEXT,
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sent_date TEXT,
            status TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

# Function to fetch stock data
def fetch_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    return data

# Function to analyze stock data
def analyze_data(data):
    recommendation = "Hold"
    if data['Close'][-1] > data['Close'][-30:].mean():
        recommendation = "Buy"
    elif data['Close'][-1] < data['Close'][-30:].mean():
        recommendation = "Sell"
    return recommendation

# Function to save user
def save_user(email, name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (email, name) VALUES (?, ?)', (email, name))
    conn.commit()
    conn.close()

# Function to save stock selection
def save_stock(ticker, user_id, selection_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO stocks (ticker, user_id, selection_date) VALUES (?, ?, ?)', (ticker, user_id, selection_date))
    conn.commit()
    conn.close()

# Function to save analysis result
def save_analysis(stock_id, recommendation, analysis_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO analysis (stock_id, recommendation, analysis_date) VALUES (?, ?, ?)', (stock_id, recommendation, analysis_date))
    conn.commit()
    conn.close()

# Function to log email sent
def log_email(user_id, sent_date, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO email_log (user_id, sent_date, status) VALUES (?, ?, ?)', (user_id, sent_date, status))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_tables()
 