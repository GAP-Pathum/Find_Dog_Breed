def train_model(model, train_data, val_data, epochs=10):
    """
    Trains the given model on the training and validation data.
    """
    print("Starting training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=1
    )
    print("Training complete.")
    return history
