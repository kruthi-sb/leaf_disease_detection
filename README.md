# Design and Implementation of AI-Powered Plant Disease Detection and Remediation

## Description

This project carries out "leaf disease detection" using **[PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)** and **[YOLOv5](https://github.com/ultralytics/yolov5)** model. The remedy recommendation is given by Gemini API, the large language model by Google. A Web-app is created using Streamlit to provide a user-friendly interface for the user to interact with the model. 

## Dataset Description

Dataset source: https://datasetninja.com/plantdoc

The PlantDoc is a dataset for plant disease detection and classification. The dataset consists of 2482 images with 8595 labeled objects belonging to 29 different classes. Images in the PlantDoc dataset have bounding box annotations. All images are labeled (i.e. with annotations). There are 2 splits in the dataset: train (2251 images) and test (231 images). The dataset was released in 2019 by the Indian Institute of Technology Gandhinagar.

## Dataset Folder Structure

The dataset is organized in the following folder structure:
```
PlantDoc
│
└───train
│   └───img
|   |   └───1img.jpg
|   |   └───2img.jpg
│   └───ann
|       └───1img.json
|       └───2img.json
└───test
    └───img
    |   └───1img.jpg
    |   └───2img.jpg
    └───ann
        └───1img.json
        └───2img.json
``` 

The annotation folder consists of json files with the following structure: (for example)

```json
{
    "description": "",
    "tags": [],
    "size": {
        "height": 2988,
        "width": 5312
    },
    "objects": [
        {
            "id": 14731013,
            "classId": 21091,
            "description": "",
            "geometryType": "rectangle",
            "labelerLogin": "gr@datasetninja.com",
            "createdAt": "2023-07-12T11:26:46.233Z",
            "updatedAt": "2023-07-12T11:26:46.233Z",
            "tags": [],
            "classTitle": "Apple rust leaf",
            "points": {
                "exterior": [
                    [
                        1690,
                        324
                    ],
                    [
                        5078,
                        2397
                    ]
                ],
                "interior": []
            }
        }
    ]
}
```

In the above json file, the classTitle is the label of the object and the points are the coordinates of the bounding box. The first 2 coordinates  represent xmin and ymin. The next 2 coordinates represent xmax and ymax.

## Methodology
NOTE:  The following 2 steps are done for both train and test datasets.

1. **Load Dataset:** 
We need to load the images in PIL format into a list. The coordinates and the labels are stored into a dictionary: {'boxes': boxes, 'labels': labels}. 
- The boxes are in the format: [xmin, ymin, xmax, ymax]. The labels are in the format: [label1, label2, label3, ...]. 
- This is done for each bounding box in the image. The ```load_dataset``` function in load_data_labels.ipynb is used for this job.

2. **Create text files from annotations:**
We need to create text files for each image. It is named after the corresponding image with .txt extension. The text file consists:
```text
class_id x_center y_center width height
```
The ```create_labels``` function does this task.
The text files are manually moved to the repective img folder. 
Now the folder structure is:
```
PlantDoc
│   
└───train
│   └───img
|   |   └───1img.jpg
|   |   └───1img.txt
|   |   └───2img.jpg
|   |   └───2img.txt
│   └───ann
|       └───1img.json
|       └───2img.json
└───test
    └───img
    |   └───1img.jpg
    |   └───1img.txt
    |   └───2img.jpg
    |   └───2img.txt
    └───ann
        └───1img.json
        └───2img.json
```
The Dataset is now ready for training.

3. **Using YOLOv5 for training:** 
The [YOLOv5 gihub repository](https://github.com/ultralytics/yolov5) is cloned. A ```data.yaml``` is created inside the data folder of yolov5.
The ```data.yaml``` file consists of the path to the train and test images, the number of classes and the names of the classes. 
```yaml
train: C:\Users\path\to\train\images
val: C:\Users\path\to\test\images

nc: 29  
names: ['Strawberry leaf', 'Peach leaf', 'Tomato leaf mosaic virus', 'Soyabean leaf', 'grape leaf', 'Tomato leaf bacterial spot', 'Bell_pepper leaf', 'Tomato leaf', 'Apple leaf', 'Apple Scab Leaf', 'Potato leaf', 'Potato leaf early blight', 'Tomato leaf yellow virus', 'Tomato Septoria leaf spot', 'Corn leaf blight', 'Potato leaf late blight', 'Bell_pepper leaf spot', 'Squash Powdery mildew leaf', 'Tomato two spotted spider mites leaf', 'Tomato mold leaf', 'Cherry leaf', 'Tomato leaf late blight', 'Apple rust leaf', 'Tomato Early blight leaf', 'Corn Gray leaf spot', 'Blueberry leaf', 'Corn rust leaf', 'grape leaf black rot', 'Raspberry leaf']
```
4. **Training and Detection:**
The ```train.py``` file is run to train the model with the following command:
```bash
!python train.py --img 640 --batch 15 --epochs 20 --data data.yaml --cfg yolov5s.yaml --name leaf_detect
```
Here:
- --img 640: The input image size is 640x640
- --batch 15: The batch size is the number of images processed in one iteration (15 images)
- --epochs 20: The number of times the model is trained on the entire dataset (20 times)
- --data data.yaml: The path to the data.yaml file
- --cfg yolov5s.yaml: The path to the configuration file of yolov5 small model
- --name leaf_detect: The name of the model (directory in which the results will be stored)

The trained model is saved in the runs folder which consists of the model weights, graphs, validation image results with bounding boxes predicted by the model.

Detection of leaf disease of a single image input is done using the following command:
```bash
!python detect.py --weights runs/train/leaf_detect/weights/best.pt --img 640 --conf 0.4 --source C:\Users\path\to\test\images
```

5. **Gemini API Integration:**
- The Gemini API is used to get the remedy recommendation for the detected leaf disease. 
- The Gemini API set up can be done by refering to the [Gemini API documentation](https://ai.google.dev/tutorials/python_quickstart).
- The necessary import ```import google.generativeai as genai``` is done.

6. **Integration of Detection results with Gemini API and creating a web interface using Streamlit:**
- Firstly, the detection on a single image is done using ```run_detection``` function in the ```Detection.py``` file.
- The latest detection results are fetched from the detect folder under runs in yolov5 directory. This is done using the ```check_latest_detection``` function.
- The ```get_response``` function is used to get the remedy recommendation from the Gemini API.
- Gemini-pro is used when the detection is successful and the label file is generated in the detect folder. The class label predicted is extracted and passed on to the API along with a text prompt to generate the remedy recommendation.
- Gemini-pro-vision is used when the detection was not successful. The image is passed on to the API along with a text prompt to generate the remedy recommendation.
- A simple web interface is built using Steamlit to interact with the model. The user can upload an image and get the detection and remedy recommendation.

## Installation

1. Clone this repository and cd into it.
2. Set up your environment using the following command:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate tf_gpu
```
4. Create a ```config.json``` file with all the necessary file and folder paths of the dataset and model (yolov5).
5. Run the app.py file using the following command:
```bash
streamlit run --server.enableCORS false --server.enableXsrfProtection false app.py
```
6. The web interface will be available at the local host address provided in the terminal. Input an image from the test folder of images and get the detection and remedy recommendation.

## References
- [PlantDoc Dataset](https://datasetninja.com/plantdoc)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Gemini API](https://ai.google.dev/tutorials/python_quickstart)
- [Youtube Tutorial](https://www.youtube.com/watch?v=mFrnRIVj8m0)

## Contributors
- [Kruthi S B](https://github.com/kruthi-sb)
- [Nitisha Patil](https://github.com/nitpat25)