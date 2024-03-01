import streamlit as st
from PIL import Image
import os
from Detect import DetectAndRespond  # Import your Backend class from your module
import json

# get image folder path from config
with open('config.json', 'r') as file:
    config = json.load(file)
    image_folder_path = config["test_img"]["img_folder_path"]

# build the UI
st.title("Plant Disease Detection and Remedy Recommendation")
st.write("Upload a picture of a plant leaf and we will tell you the disease and recommend a remedy for it")

uploaded_file = st.file_uploader("Choose a plant leaf image...", type="jpg")

if uploaded_file is not None:

    # display the image
    # get the file name
    file_name = uploaded_file.name
    st.image(uploaded_file, caption=file_name, use_column_width=True)
    st.write(" ## Detecting...")

    # create an instance of the Backend class
    # give path to the uploaded file
    images_path = image_folder_path
    file_path = images_path + file_name
    st.write(file_path)
    backend = DetectAndRespond(file_path)
    response = backend.run_all()

    # display the result
    st.write(response)