import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# SMTP server configuration
smtp_server = 'localhost'
smtp_port = 1025
sender_email = 'nejlayvazoglu@gmail.com'
receiver_email = 'nejlaayvazoglu@hotmail.com'

# Create a multipart message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = 'Test Email'

# Add body to email
body = 'This is a test email sent using Python.'
msg.attach(MIMEText(body, 'plain'))

try:
    # Connect to SMTP server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        # Send email
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print('Email sent successfully!')

except Exception as e:
    print(f'Error sending email: {e}')