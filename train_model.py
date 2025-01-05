import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import mlflow.keras
import pandas as pd
import joblib  # For saving the scaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv("processed_data.csv", index_col="timestamp", parse_dates=True)

# Selected features (excluding 'pm2')
features = ["aqi", "temperature", "pressure", "humidity", "wind_speed"]
print(data[features].info())
print(data.head())

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Save the scaler
joblib.dump(scaler, "model/scaler.pkl")

# Create time-series dataset
def create_dataset(dataset, time_steps=10):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i + time_steps, :])
        y.append(dataset[i + time_steps, 0])  # Predict 'aqi'
    return np.array(X), np.array(y)
scaled_data.reshape(-1, len(features))
time_steps = 10
X, y = create_dataset(scaled_data, time_steps)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, len(features))),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# MLflow Experiment
mlflow.set_experiment("Pollution Trend Prediction")
with mlflow.start_run():
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32
    )

    # Log the model with MLflow
    mlflow.keras.log_model(model, "model")

    # Log metrics
    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])
    mlflow.log_metric("train_mae", history.history["mae"][-1])
    mlflow.log_metric("val_mae", history.history["val_mae"][-1])

# Predictions and evaluation
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")

# Log test metrics to MLflow
with mlflow.start_run():
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
