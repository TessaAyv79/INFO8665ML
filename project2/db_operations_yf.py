# hnfgi.py

import ibm_db
import streamlit as st
import pandas as pd

# Db2 connection details
dsn_hostname = "your_hostname"
dsn_uid = "your_username"
dsn_pwd = "your_password"
dsn_driver = "{IBM DB2 ODBC DRIVER}"
dsn_database = "your_database"
dsn_port = "50000"
dsn_protocol = "TCPIP"
dsn = (
    f"DATABASE={dsn_database};"
    f"HOSTNAME={dsn_hostname};"
    f"PORT={dsn_port};"
    f"PROTOCOL={dsn_protocol};"
    f"UID={dsn_uid};"
    f"PWD={dsn_pwd};"
)

def connect_to_db2():
    try:
        conn = ibm_db.connect(dsn, "", "")
        st.write("Connected to Db2 database")
        return conn
    except Exception as e:
        st.error(f"Unable to connect to the database: {e}")
        return None

def create_table(conn):
    create_sql = """
    CREATE TABLE stock_data (
        Date DATE,
        Open FLOAT,
        High FLOAT,
        Low FLOAT,
        Close FLOAT,
        Volume BIGINT,
        Dividends FLOAT,
        Stock_Splits FLOAT
    )
    """
    try:
        ibm_db.exec_immediate(conn, create_sql)
        st.write("Table created")
    except Exception as e:
        st.error(f"Failed to create table: {e}")

def save_data_to_db2(conn, df, table_name):
    columns = ", ".join(df.columns)
    placeholders = ", ".join(["?" for _ in df.columns])
    insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        stmt = ibm_db.prepare(conn, insert_sql)
        for index, row in df.iterrows():
            ibm_db.execute(stmt, tuple(row))
        st.write(f"Data saved to {table_name}")
    except Exception as e:
        st.error(f"Failed to save data: {e}")

def load_data_from_db2(conn, table_name):
    select_sql = f"SELECT * FROM {table_name}"
    try:
        stmt = ibm_db.exec_immediate(conn, select_sql)
        rows = []
        row = ibm_db.fetch_assoc(stmt)
        while row:
            rows.append(row)
            row = ibm_db.fetch_assoc(stmt)
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()