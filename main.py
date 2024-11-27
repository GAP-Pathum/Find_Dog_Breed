#main.py
import os
from src.data_loader import load_data
from src.model import build_model
from src.train import train_model
from src.predict import load_saved_model, predict_breed

# Paths
DATA_DIR = "data/StanfordDogs"
MODEL_PATH = "models/saved_model.h5"
TEST_IMAGE = r"C:\Users\pathu\Desktop\1.jpg"  # Path to your test image

if __name__ == "__main__":
    print("Starting the Dog Breed Identification project...")

    # Check if the saved model exists
    if os.path.exists(MODEL_PATH):
        print(f"Model found at {MODEL_PATH}. Skipping training...")
        model = load_saved_model(MODEL_PATH)
    else:
        print("No saved model found. Proceeding with training...")

        # Step 1: Load and preprocess data
        train_data, val_data = load_data(DATA_DIR)

        # Step 2: Build the model
        print("Building the model...")
        model = build_model()

        # Step 3: Train the model
        print("Training the model...")
        history = train_model(model, train_data, val_data)

        # Step 4: Save the model
        print(f"Saving the model to {MODEL_PATH}...")
        os.makedirs("models", exist_ok=True)
        model.save(MODEL_PATH)
        print("Model saved successfully!")

    # Step 5: Load class indices for predictions
    print("Loading class indices...")
    train_data, _ = load_data(DATA_DIR)  # Load data to access class indices
    class_indices = train_data.class_indices  # Get mapping of class labels to indices

    # Predict the breed of a test image
    print(f"Making prediction for {TEST_IMAGE}...")
    predicted_breed = predict_breed(TEST_IMAGE, model, class_indices)
    print(f"The predicted breed is: {predicted_breed}")
