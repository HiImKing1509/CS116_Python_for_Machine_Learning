import streamlit as st
from PIL import Image
import cv2
import numpy as np
image = Image.open('image_1.jpg')

streamlit_style = """
	<style>
		@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

		html, body, [class*="css"]  {
		    font-family: 'Roboto', sans-serif;
		}
	</style>
"""
st.markdown(streamlit_style, unsafe_allow_html=True)
st.markdown(
    """
        # Name: Huynh Viet Tuan Kiet
        # ID: 20521494
    """
)

a_value = st.text_input("Enter value 1:")
b_value = st.text_input("Enter value 2:")
operator_option = st.selectbox("Choose operator", ["Plus", "Minus", "Divide", "Multiply"]) #Can change radio buttons

button = st.button("Calc")

if button:
    if operator_option == "Plus":
        st.text_input("Result:", float(a_value) + float(b_value))
    if operator_option == "Minus":
        st.text_input("Result:", float(a_value) - float(b_value))
    if operator_option == "Divide":
        st.text_input("Result:", float(a_value) / float(b_value))
    if operator_option == "Multiply":
        st.text_input("Result:", float(a_value) * float(b_value))
    
st.image(image, caption='Sunrise by the mountains')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    img_path = './' + uploaded_file.name
    st.write(bytes_data)
    with open(img_path, 'wb') as f:
        f.write(bytes_data)
    img = cv2.imread(img_path, 0)
    
    # Filter images
    filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]) 
    result1 = cv2.filter2D(img, -1, filter)
    
    filter = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]) 
    result2 = cv2.filter2D(img, -1, filter)
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Result filter 1")
        st.image(result1)
    with col2:
        st.header("Result filter 2")
        st.image(result2)