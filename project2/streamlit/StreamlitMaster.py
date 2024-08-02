import streamlit as st
from MailMagic import send_email  # Import your send_email function from MailMagic.py

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title('Stock Analysis and Forecasting App')

    # Example: Get user input for selected stock
    selected_stock = st.sidebar.selectbox('Select a stock', ['AAPL', 'GOOGL', 'MSFT'])

    # Collect user input for email and password
    receiver_email = st.sidebar.text_input('Enter your email')
    sender_email = st.sidebar.text_input('Enter your sender email')  # Add sender email input
    sender_password = st.sidebar.text_input('Enter your password', type='password')  # Securely collect password

    # Example: Send email on forecast update
    if st.sidebar.checkbox('Send Forecast Update Email'):
        if st.sidebar.button('Send Email'):
            subject = f"Stock Forecast Update for {selected_stock}"
            message = f"Dear user,\n\nHere is the latest forecast update for {selected_stock}."
            # Pass all necessary parameters to send_email function
            send_email(sender_email, sender_password, receiver_email, subject, message)
            st.sidebar.success(f"Email sent successfully to {receiver_email}!")

    # Remainder of your Streamlit app code

if __name__ == '__main__':
    main()