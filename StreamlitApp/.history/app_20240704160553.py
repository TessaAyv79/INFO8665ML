import streamlit as st
# Example: Displaying an image in Streamlit

# Set the file path to the image
image_path = "C:\\Users\\Admin\\Documents\\MLAI\\INFO8665ML\\StreamlitApp\\images2.jpg"

# Display the image in the app
st.image(image_path, caption='Your Image Caption', use_column_width=True)