import streamlit as st

# Title of the app
st.title('My Streamlit App')

# Display the image
image_path = "C:\\Users\\Admin\\Documents\\MLAI\\INFO8665ML\\StreamlitApp\\images2.jpg"
st.image(image_path, caption='Image Description Here', use_column_width=True)

# Additional Streamlit app content
st.write("Welcome to my Streamlit app. This is where you can add more content.")