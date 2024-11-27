import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_saved_model(model_path):
    """
    Load the saved model from the specified path.
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model

def predict_breed(image_path, model, class_indices):
    """
    Predict the breed of the dog in the given image.
    """
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))  # Resize to match model input
    image_array = img_to_array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping
    predicted_label = class_labels[predicted_class]

    return predicted_label
