from smtpd import SMTPServer

class CustomSMTPServer(SMTPServer):
    def process_message(self, peer, mailfrom, rcpttos, data, **kwargs):
        print(f"Received message from: {mailfrom}, to: {rcpttos}")

if __name__ == "__main__":
    HOST = 'localhost'
    PORT = 1025

    print(f"Starting dummy SMTP server on {HOST}:{PORT}")
    server = CustomSMTPServer((HOST, PORT), None)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the server.")
        server.close()