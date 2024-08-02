from aiosmtpd.controller import Controller
import asyncio

class CustomSMTPHandler:
    async def handle_DATA(self, server, session, envelope):
        print(f"Received message from: {envelope.mail_from}, to: {envelope.rcpt_tos}")
        return '250 Message accepted for delivery'

if __name__ == "__main__":
    HOST = 'localhost'
    PORT = 1025

    handler = CustomSMTPHandler()

    controller = Controller(handler, hostname=HOST, port=PORT)

    try:
        print(f"Starting dummy SMTP server on {HOST}:{PORT}")
        asyncio.run(controller.start())
    except KeyboardInterrupt:
        print("\nShutting down the server.")
        asyncio.run(controller.stop())