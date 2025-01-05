from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
import numpy as np
import joblib
import os
import requests
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Summary, Counter, Histogram, generate_latest, REGISTRY

# Load the Keras model
model = tf.keras.models.load_model("model/best_model.h5", custom_objects={"mse": "mean_squared_error"})

# Load the scaler
scaler = joblib.load("model/scaler.pkl")

# Flask app
app = Flask(__name__)

# Start Prometheus HTTP server
start_http_server(8000)

# Create Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTION_TIME = Histogram('prediction_processing_seconds', 'Time spent making predictions')
TOTAL_REQUESTS = Counter('total_requests', 'Total number of requests')
FAILED_REQUESTS = Counter('failed_requests', 'Total number of failed requests')

# API to fetch live pollution and weather data
AIRVISUAL_API_KEY = os.getenv("AIRVISUAL_API_KEY")
url = f"https://api.airvisual.com/v2/nearest_city?key={AIRVISUAL_API_KEY}"


def fetch_live_data(lat, lon):
    """Fetch live air pollution and weather data."""
    pollution_response = requests.get(url)
    pollution_response.raise_for_status()

    if pollution_response.status_code == 200:
        pollution_data = pollution_response.json()
        weather_data = pollution_data["data"]["current"]["weather"]
        return pollution_data, weather_data
    else:
        raise Exception("Failed to fetch live data. Check your API key or network connection.")


@PREDICTION_TIME.time()
def make_predictions(current_data):
    """Scale inputs and make predictions."""
    
    # Reshape current_data to (1, 1, 5) (1 batch, 1 time step, 5 features)
    current_data = current_data.reshape((1, 1, 5))
    print(f"Shape after reshape (1, 1, 5): {current_data.shape}")

    # Repeat this data for 10 time steps (axis=1)
    current_data = np.repeat(current_data, 10, axis=1)  # Shape will now be (1, 10, 5)
    print(f"Shape after repeat (1, 10, 5): {current_data.shape}")

    # Scale inputs (reshape the data to 2D before scaling)
    # Reshape to (samples, features) i.e., (1*10, 5)
    current_data_2d = current_data.reshape((-1, 5))  # Shape will be (10, 5)
    scaled_inputs = scaler.transform(current_data_2d)  # Perform scaling

    # Now reshape back to 3D for LSTM
    scaled_inputs = scaled_inputs.reshape((1, 10, 5))  # Shape (1, 10, 5)

    print(f"Shape of scaled inputs after transformation: {scaled_inputs.shape}")

    # Now you can pass the scaled data into the model for prediction
    return model.predict(scaled_inputs)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
@REQUEST_TIME.time()
def predict():
    try:
        # Increment total requests counter
        TOTAL_REQUESTS.inc()

        # Get location from query params or use default
        lat = request.args.get("lat", 33.6844)  # Default: Islamabad latitude
        lon = request.args.get("lon", 73.0479)  # Default: Islamabad longitude

        # Fetch live data
        pollution_data, weather_data = fetch_live_data(lat, lon)

        # Prepare input features (assuming specific fields)
        current_data = np.array([[
            pollution_data["data"]["current"]["pollution"]["aqius"],  # Current AQI
            weather_data["tp"],  # Temperature
            weather_data["pr"],  # Pressure
            weather_data["hu"],  # Humidity
            weather_data["ws"],  # Wind Speed
        ]])

        # # Reshape current_data to (1, 1, 5) (1 batch, 1 time step, 5 features)
        # current_data = current_data.reshape((1, 1, 5))
        # print(f"Shape after reshape (1, 1, 5): {current_data.shape}")

        # # Repeat this data for 10 time steps (axis=1)
        # current_data = np.repeat(current_data, 10, axis=1)  # Shape will now be (1, 10, 5)
        # print(f"Shape after repeat (1, 10, 5): {current_data.shape}")

        

        # Predict AQI for the next 24 hours
        predictions = make_predictions(current_data)
        x= predictions.flatten().tolist()
        print(x,"hello")
        
        input_data_for_inverse_transform = [[x[0], 0, 0, 0, 0]]
        predicted_aqi = scaler.inverse_transform(input_data_for_inverse_transform)[0][0]
        print(f"Predicted AQI for 1 hour ahead: {predicted_aqi:.2f}")

        response = {
            "location": f"{pollution_data['data']['city']}, {pollution_data['data']['country']}",
            "current_aqi": pollution_data["data"]["current"]["pollution"]["aqius"],
            "temperature": weather_data["tp"],
            "predictions": [predicted_aqi]
        }
        return jsonify(response)
    except Exception as e:
        FAILED_REQUESTS.inc()
        return jsonify({"error": str(e)}), 400


@app.route("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
