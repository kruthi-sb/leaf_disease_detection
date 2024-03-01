import os
import google.generativeai as genai
import textwrap
from IPython.display import display
from IPython.display import Markdown
import json
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

class DetectAndRespond:
    def __init__(self, img_path):
        self.img_path = img_path
        with open('C:\\Users\\kruth\\leaf_disease_detection\\config.json') as f:
            self.config = json.load(f)
        self.yolov5_path = self.config['yolov5']["yolov5"]
        self.weights_path = self.config["yolov5"]['weights']
        self.class_dict = {'Strawberry leaf': 0, 'Peach leaf': 1, 'Tomato leaf mosaic virus': 2, 'Soyabean leaf': 3, 'grape leaf': 4, 'Tomato leaf bacterial spot': 5, 'Bell_pepper leaf': 6, 'Tomato leaf': 7, 'Apple leaf': 8, 'Apple Scab Leaf': 9, 'Potato leaf': 10, 'Potato leaf early blight': 11, 'Tomato leaf yellow virus': 12, 'Tomato Septoria leaf spot': 13, 'Corn leaf blight': 14, 'Potato leaf late blight': 15, 'Bell_pepper leaf spot': 16, 'Squash Powdery mildew leaf': 17, 'Tomato two spotted spider mites leaf': 18, 'Tomato mold leaf': 19, 'Cherry leaf': 20, 'Tomato leaf late blight': 21, 'Apple rust leaf': 22, 'Tomato Early blight leaf': 23, 'Corn Gray leaf spot': 24, 'Blueberry leaf': 25, 'Corn rust leaf': 26, 'grape leaf black rot': 27, 'Raspberry leaf': 28}
    
    def run_detection(self):
        st.write("Running Detection...")
        # Change directory to yolov5 folder
        os.chdir(self.yolov5_path)

        command = [
            'python', 'detect.py',
            '--source', self.img_path,
            '--weights', self.weights_path,
            '--img', '416',
            '--conf', '0.5',
            '--iou', '0.4',
            '--save-txt',
            '--save-conf'
        ]
        # Run the command and capture output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Decode output bytes to strings and display
        print("Running Detection...")
        print(stdout.decode())
        print(stderr.decode())

    def check_latest_detection(self):

        # Change directory to yolov5 folder and get the path of detect folder
        os.chdir(self.yolov5_path)
        detect_folder = os.path.join('runs', 'detect')

        # Get the list of subdirectories in the detect folder
        subdirs = [d for d in os.listdir(detect_folder) if os.path.isdir(os.path.join(detect_folder, d))]

        # Sort subdirectories by creation time to get the latest one
        latest_folder = sorted(subdirs, key=lambda x: os.path.getctime(os.path.join(detect_folder, x)), reverse=True)[0]
        
        # Get the path of the latest detection folder
        latest_folder_path = os.path.join(detect_folder, latest_folder)
        labels_folder_path = os.path.join(latest_folder_path, "labels")

        # Get the image path from the latest folder in detect folder
        image_path = os.path.join(latest_folder_path, os.listdir(latest_folder_path)[0])
        # store image using PIL
        image = Image.open(image_path)
        # display the image using plt
        #plt.imshow(image)
        #plt.axis('off')
        # display the image using streamlit
        st.write("Detected Image")
        st.image(image, caption="Detected Image", use_column_width=True)

        # Check if labels folder exists
        if os.path.exists(labels_folder_path):
            print("Detection successful!")
            st.write("Detection successful!")

            # Get the path of the labels file
            labels_file_path = os.path.join(labels_folder_path, os.listdir(labels_folder_path)[0])
            
            # Read the content of the labels file
            with open(labels_file_path, 'r') as file:
                content = file.readline()
            
            # Extract the class index from the content
            class_index = content.split()[0]
            #print("Class index:", class_index)

            # Get the class name from the class index
            class_name = list(self.class_dict.keys())[list(self.class_dict.values()).index(int(class_index))]
            print("Class name:", class_name)
            st.write(" # Class name:", class_name)
            
            return image, class_name
        # if labels folder does not exist, return image and None
        else:
            print("Labels folder does not exist in the latest detection folder.")
            st.write("Labels folder does not exist in the latest detection folder.")
            return image, None
    
    # utility function to turn response into markdown content
    # turn response into markdown content
    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
        
    def get_response(self):

        # get class_name and image from check_latest_detection
        image, class_name = self.check_latest_detection()
        st.write("Getting Response...")

        # if class_name exists, get response from gemini-pro model
        if class_name:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"I want the information on {class_name}. If it is a disease, then provide remedies to cure the disease."
            response = model.generate_content(prompt)
            #self.to_markdown(response.text)
            print(response.text)
        
        # if detection is not successful, get response from gemini-vision-pro model
        else:
            model = genai.GenerativeModel('gemini-vision-pro')
            prompt = "Detect the type of plant and the disease if any. If it is a disease, then provide remedies to cure the disease."
            response = model.generate_content([prompt, image])
            #self.to_markdown(response.text)
            print(response.text)
        
        return response.text
    
    def run_all(self):
        self.run_detection()
        response = self.get_response()
        return response