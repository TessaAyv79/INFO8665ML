import os
from dotenv import load_dotenv
import streamlit as st
ap
key = "my-secret-api-key"
print("This is a sample Python script.")

def main():
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    
    print(f"Connecting to database at {db_host}:{db_port} with user {db_user}")

if __name__ == "__main__":
    main()