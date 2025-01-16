# Face Detection App

This project is a face detection application built using the YOLOv8 model and Streamlit. The app detects faces in uploaded images and displays the number of detected faces along with the image containing bounding boxes around the faces.

## Description
This application leverages advanced deep learning models to perform real-time face detection in images. The intuitive Streamlit interface allows users to easily upload images and visualize detection results with bounding boxes drawn around faces. It is an ideal starting point for exploring computer vision applications in Python.

## Features
- Upload an image in JPG, JPEG, or PNG format.
- Detect faces in the uploaded image using the YOLOv8 model.
- Display the number of faces detected.
- Show the image with bounding boxes drawn around detected faces.

## Requirements
- Python 3.8+
- Libraries:
  - `streamlit`
  - `huggingface-hub`
  - `ultralytics`
  - `supervision`
  - `pillow`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Shah114/Face-Detection-App.git
   cd face-detection-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
