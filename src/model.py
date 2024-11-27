# src/model.py
import tensorflow as tf

def build_model():
    """
    Builds and compiles a CNN model using MobileNetV2 as the base.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model layers

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),  # Regularization
        tf.keras.layers.Dense(120, activation='softmax')  # Adjust for 120 breeds
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
