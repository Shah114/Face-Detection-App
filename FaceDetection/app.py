# Importing Modules
import streamlit as st
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageDraw
import numpy as np

# Load the model from Hugging Face hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    return YOLO(model_path)

# Draw bounding boxes on the image
def draw_bounding_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_min, y_min, x_max, y_max = box[:4]
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
    return image

# Main Streamlit app
def main():
    st.title("Face Detection App")
    st.write("Upload an image, and the app will detect faces and display the number of detected faces.")

    # Load YOLO model
    model = load_model()

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform inference
        output = model(image)
        results = Detections.from_ultralytics(output[0])

        # Extract bounding boxes
        boxes = results.xyxy  # Coordinates of bounding boxes

        # Draw bounding boxes on the image
        image_with_boxes = draw_bounding_boxes(image.copy(), boxes)

        # Display the result
        st.image(image_with_boxes, caption="Detected Faces", use_column_width=True)

        # Display the number of detected faces
        st.write(f"Number of faces detected: {len(boxes)}")

if __name__ == "__main__":
    main()