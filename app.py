import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from src.predict import load_saved_model, predict_breed

# Paths
MODEL_PATH = "models/saved_model.h5"
DATA_DIR = "data/StanfordDogs"

# Load the saved model
model = load_saved_model(MODEL_PATH)

# Load class indices (ensure these match your training data's class indices)
class_indices = {
    0: 'Golden_Retriever',
    1: 'Labrador',
    2: 'Bulldog',  # Example: Add all your class names here (120 breeds in total)
    3: 'Poodle',
    4: 'Beagle',
    # Add more breeds...
}

# Reverse mapping of class indices to breed names
class_labels = {v: k for k, v in class_indices.items()}

# Streamlit UI
st.title("Dog Breed Identifier")
st.write("Upload an image or use your camera to identify the dog's breed!")

# Upload Image Section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process and make prediction
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    predicted_label = class_indices[predicted_class]  # Map index to breed name

    st.write(f"**Predicted Breed:** {predicted_label}")

# Use Camera Section
use_camera = st.checkbox("Use Camera")

if use_camera:
    # Use Streamlit's camera input widget
    camera_capture = st.camera_input("Take a picture")

    if camera_capture:
        # Convert the camera image to a format that can be processed
        image = load_img(camera_capture, target_size=(224, 224))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
        predicted_label = class_indices[predicted_class]  # Map index to breed name

        st.write(f"**Predicted Breed:** {predicted_label}")

# Error handling for missing or incorrect inputs
if uploaded_file is None and not use_camera:
    st.warning("Please upload an image or use the camera to make a prediction.")
