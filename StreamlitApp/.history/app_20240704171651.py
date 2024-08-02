import streamlit as st

# Title of the app
st.title('My Streamlit App')

# Display the image
diabetes_model = pickle.load(open('/app/saved_models/diabetes_model.sav', 'rb'))

# Additional Streamlit app content
st.write("Welcome to my Streamlit app. This is where you can add more content.")