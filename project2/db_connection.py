import ibm_db
import streamlit as st

def connect_to_db2(dsn_hostname, dsn_uid, dsn_pwd, dsn_database, dsn_port="50000", dsn_protocol="TCPIP"):
    dsn_driver = "{IBM DB2 ODBC DRIVER}"
    dsn = (
        f"DATABASE={dsn_database};"
        f"HOSTNAME={dsn_hostname};"
        f"PORT={dsn_port};"
        f"PROTOCOL={dsn_protocol};"
        f"UID={dsn_uid};"
        f"PWD={dsn_pwd};"
    )

    try:
        conn = ibm_db.connect(dsn, "", "")
        st.write("Connected to Db2 database")
        return conn
    except Exception as e:
        st.error(f"Unable to connect to the database: {e}")
        return None