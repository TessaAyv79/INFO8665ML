import socket

# Define the SMTP server address and port
smtp_server = 'localhost'
smtp_port = 1025

# Create a socket connection to the SMTP server
try:
    smtp_socket = socket.create_connection((smtp_server, smtp_port))
    print(f"Connected to SMTP server {smtp_server} on port {smtp_port}")
    smtp_socket.close()
except Exception as e:
    print(f"Error connecting to SMTP server: {e}")