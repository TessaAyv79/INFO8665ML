from aiosmtpd.controller import Controller
import asyncio

class CustomSMTPHandler:
    async def handle_DATA(self, server, session, envelope, **kwargs):
        print(f"Received message from: {envelope.mail_from}, to: {envelope.rcpt_tos}")
        return '250 Message accepted for delivery'

async def start_smtp_server():
    HOST = 'localhost'
    PORT = 1025

    handler = CustomSMTPHandler()
    controller = Controller(handler, hostname=HOST, port=PORT)

    try:
        print(f"Starting dummy SMTP server on {HOST}:{PORT}")
        loop = asyncio.get_event_loop()
        server_task = loop.create_task(controller.start())
        await server_task
    except KeyboardInterrupt:
        print("\nShutting down the server.")
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(start_smtp_server())