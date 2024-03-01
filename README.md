# Design and Implementation of AI-Powered Plant Disease Detection and Remediation

## Description

This project carries out "leaf disease detection" using **[PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)** and **[YOLOv5](https://github.com/ultralytics/yolov5)** model. The remedy recommendation is given by Gemini API, the large language model by Google. A Web-app is created using Streamlit to provide a user-friendly interface for the user to interact with the model. 

## Dataset Description

Dataset source: https://datasetninja.com/plantdoc

The PlantDoc is a dataset for plant disease detection and classification. The dataset consists of 2482 images with 8595 labeled objects belonging to 29 different classes. Images in the PlantDoc dataset have bounding box annotations. All images are labeled (i.e. with annotations). There are 2 splits in the dataset: train (2251 images) and test (231 images). The dataset was released in 2019 by the Indian Institute of Technology Gandhinagar.

## Dataset Folder Structure

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

The annotation folder consists of json files with the following structure:

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


