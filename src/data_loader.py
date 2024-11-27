import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    """
    Load and preprocess data using ImageDataGenerator.
    """
    print(f"Loading data from {data_dir}...")

    # Create ImageDataGenerator for training and validation
    datagen = ImageDataGenerator(
        rescale=1.0/255,          # Normalize pixel values
        validation_split=0.2      # 20% data for validation
    )

    # Training data
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),   # Resize images to 224x224
        batch_size=32,
        class_mode='sparse',
        subset='training'
    )

    # Validation data
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        subset='validation'
    )

    return train_generator, val_generator
