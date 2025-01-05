
# Environmental Monitoring and Pollution Prediction System  

## Project Overview  
This project implements a robust pipeline to collect, process, and predict environmental pollution trends. The system leverages APIs like **AirVisual** and **OpenWeather** to collect weather and pollution data. A **Long Short-Term Memory (LSTM)** model is trained to forecast **Air Quality Index (AQI)** trends, and predictions are served through a **Flask API**. Real-time system monitoring and performance tracking are implemented using **Prometheus** and **Grafana**.  

---

## Features  
1. **Data Collection**:  
   - Fetches weather and pollution data from APIs (e.g., AirVisual, OpenWeather).  
   - Automated data collection using Python scripts.  
   - Dataset versioning with **DVC** to ensure reproducibility.  

2. **Model Training**:  
   - LSTM-based model for time-series AQI prediction.  
   - Hyperparameter tuning using **Random Search**.  
   - Experiment tracking and model logging using **MLflow**.  

3. **Deployment**:  
   - Flask-based API for serving real-time predictions.  
   - `/predict`: Endpoint for location-based AQI predictions.  
   - `/metrics`: Prometheus metrics for system monitoring.  

4. **Monitoring**:  
   - **Prometheus** for collecting metrics like request latency, API performance, and errors.  
   - **Grafana** for real-time visualization and alert configuration.  

---

## Dataset Details  
The dataset was generated during Task 1 using the following sources:  
- **AirVisual API**: Provides AQI data and pollutant concentrations.  
- **OpenWeather API**: Supplies weather data, including temperature, pressure, and humidity.  

### Features:  
1. `aqi`: Air Quality Index (target variable).  
2. `temperature`: Atmospheric temperature in °C.  
3. `pressure`: Barometric pressure in hPa.  
4. `humidity`: Relative humidity in %.  
5. `wind_speed`: Wind speed in m/s.  

---

## Model Training and Development  

### Model Architecture:  
The model is an LSTM-based time-series predictor. Key components:  
- **Input Shape**: Sequences of 10-time steps, each with 5 features.  
- **LSTM Layers**:  
  - First layer: 50 units with return sequences enabled.  
  - Second layer: 50 units without return sequences.  
- **Dense Layer**: Single output predicting the next AQI value.  
- **Loss Function**: Mean Squared Error (MSE).  
- **Optimizer**: Adam.  

### Hyperparameter Tuning:  
Random Search was used to optimize the following parameters:  
- **LSTM Units**: [32, 50, 64, 100].  
- **Batch Size**: [8, 16, 32].  
- **Learning Rate**: [0.001, 0.0001].  
- **Epochs**: [5, 10, 15, 20].  

### Best Configuration:  
- **LSTM Units**: 64.  
- **Batch Size**: 8.  
- **Learning Rate**: 0.001.  
- **Epochs**: 10.  

**Best Validation Loss**: 0.077.  
**Best Validation MAE**: 0.217.  

---

## Deployment  

### Flask API:  
The trained LSTM model was deployed using Flask.  
- **Endpoints**:  
  1. `/predict`: Accepts query parameters (`lat`, `lon`) to fetch live weather and pollution data and generate predictions.  
  2. `/metrics`: Exposes system performance metrics for Prometheus.  

**Example Prediction Workflow**:  
1. Fetch live data using the **AirVisual API**.  
2. Preprocess inputs (scaling and reshaping).  
3. Use the LSTM model to predict AQI for the next hour.  
4. Return predictions in JSON format.  

---

## Monitoring and Live Testing  

### Prometheus:  
- Configured to scrape system metrics from the `/metrics` endpoint.  
- Metrics include:  
  - Total requests (`total_requests`).  
  - Failed requests (`failed_requests`).  
  - Request latency (`request_processing_seconds`).  
  - Prediction processing time (`prediction_processing_seconds`).  

### Grafana:  
- Visualizes metrics collected by Prometheus.  
- Key Dashboards:  
  - API throughput and latency.  
  - System errors and uptime.  
- Alerts are set up for:  
  - High API latency.  
  - Increased error rates.  

### Live Data Testing:  
- Live weather and pollution data were fetched every hour using the **AirVisual API**.  
- Predictions were validated against actual AQI trends.  
- Average API response time: 250ms.  
- Model inference time: 20ms.  

---

## Evaluation  

### Model Performance:  
- **Root Mean Squared Error (RMSE)**: 7.8.  
- **Mean Absolute Error (MAE)**: 4.6.  

### System Performance:  
- Successfully handled 50 requests/second during stress testing.  
- Grafana dashboards showed 99.9% uptime during the testing phase.  

---

## Directory Structure  
```
project/
├── data/                         # Collected environmental data
│   └── (JSON files from APIs)
├── model/                        # Model and scaler
│   ├── best_model.h5             # Best trained LSTM model
│   └── scaler.pkl                # Scaler for input preprocessing
├── fetch_data.py                 # Script to fetch live data
├── preprocess_data.py            # Data preprocessing script
├── train_model.py                # LSTM training with MLflow integration
├── hyperparameter_tuning.py      # Script for hyperparameter tuning
├── app.py                        # Flask API for predictions
├── prometheus.yml                # Prometheus configuration file
├── docker-compose.yml            # Docker Compose for Prometheus and Grafana
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Setup Instructions  

### Prerequisites:  
1. Python (3.8 or later).  
2. Docker and Docker Compose.  
3. API Keys for **AirVisual** and **OpenWeather**.  

### Installation:  
1. Clone the repository:
   ```
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure `.env` file with API keys:
   ```
   AIRVISUAL_API_KEY=your_airvisual_key
   OPENWEATHER_API_KEY=your_openweather_key
   ```

---

### Running the Project  

#### 1. Data Collection:  
Run the script to fetch and save live data:
```
python fetch_data.py
```

#### 2. Model Training:  
Train the LSTM model:
```
python train_model.py
```

Perform hyperparameter tuning:
```
python hyperparameter_tuning.py
```

#### 3. Start Flask API:  
Serve the API:
```
python app.py
```

#### 4. Start Prometheus and Grafana:  
Run the monitoring stack:
```
docker-compose up -d
```

Prometheus: http://localhost:9090  
Grafana: http://localhost:3000 (Default credentials: admin/admin)  

---

## Future Improvements  
1. Incorporate additional features like pollutant concentrations (e.g., PM2.5, NO2).  
2. Use advanced models like hybrid LSTM + Attention for improved accuracy.  
3. Scale the API deployment using cloud services like AWS or GCP.  

---

**Authors**:  
Ruhail Rizwan  
Contact: ruhail.rizwan@example.com  
```  
