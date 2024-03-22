# Image Prediction Flask API

This project is a Flask application that serves as a REST API endpoint. It accepts an image file via a POST request and returns predictions with associated metadata in JSON format. The application uses a pre-trained machine learning model for image prediction.

## Features

- Accepts image file uploads and processes them for predictions.
- Uses TensorFlow/Keras for image-based machine learning predictions.
- Parses and returns relevant metadata for each prediction.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- Python 3
- Flask
- TensorFlow
- PIL (Pillow)
- NumPy
- Joblib
- Sklearn

## Training the model

Please get the dataset from the repo listed at the bottom,


```python training.py```


You can find the dataset used for training at https://github.com/google-research-datasets/Nutrition5k
