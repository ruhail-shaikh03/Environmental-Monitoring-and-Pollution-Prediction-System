import os
import numpy as np
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import mlflow.keras
import joblib
from train_model import create_dataset, scaled_data, X_train, y_train, X_test, y_test  # Ensure these are imported correctly

# Define the time_steps
time_steps = 10

# Parameter grid for hyperparameter tuning
param_grid = {
    "units": [32, 50, 64, 100],  # Number of LSTM units
    "batch_size": [8, 16, 32],  # Batch size
    "learning_rate": [0.001, 0.0001],  # Learning rate
    "epochs": [5, 10, 15, 20]  # Number of epochs
}

# Generate random combinations of hyperparameters
param_combinations = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

# Function to build the LSTM model
def build_model(units, learning_rate, time_steps, input_dim):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(time_steps, input_dim)),
        LSTM(units),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

# Initialize variables to track the best model
best_model = None
best_val_loss = float("inf")
best_params = {}

# Create a directory for saving the best model
os.makedirs("model", exist_ok=True)

# Run hyperparameter tuning
for params in param_combinations:
    with mlflow.start_run():
        print(f"Training with params: {params}")

        # Build the model with the current set of parameters
        model = build_model(
            units=params["units"],
            learning_rate=params["learning_rate"],
            time_steps=time_steps,
            input_dim=X_train.shape[2]
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0  # Suppress detailed logs for cleaner output
        )

        # Log hyperparameters and performance metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            "val_loss": history.history["val_loss"][-1],
            "val_mae": history.history["val_mae"][-1]
        })
        mlflow.keras.log_model(model, "model")

        # Track the best model
        val_loss = history.history["val_loss"][-1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_params = params

# Save the best model and its parameters
if best_model:
    best_model.save("model/best_model.h5")
    joblib.dump(best_params, "model/best_params.pkl")
    print("Best model saved in 'model/best_model.h5' with parameters:")
    print(best_params)
