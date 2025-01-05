import os
import json
import pandas as pd
from glob import glob


# Load AirVisual data from JSON files
def load_airvisual_data(directory):
    files = glob(os.path.join(directory, "*.json"))
    data = []
    
    for file in files:
        with open(file, "r") as f:
            content = json.load(f)
            if "data" in content:  # Ensure it's an AirVisual file
                pollution = content["data"]["current"]["pollution"]
                weather = content["data"]["current"]["weather"]
                data.append({
                    "timestamp": pollution["ts"],
                    "aqi": pollution["aqius"],
                    #"pm2_5": pollution["mainus"],
                    "temperature": weather["tp"],
                    "pressure": weather["pr"],
                    "humidity": weather["hu"],
                    "wind_speed": weather["ws"],
                })
    return pd.DataFrame(data)


# Clean and preprocess AirVisual data
def preprocess_airvisual_data(df):
    # Convert timestamps to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Sort by timestamp and set as index
    df = df.sort_values("timestamp").set_index("timestamp")
    
    # Fill missing values with forward fill, then backward fill
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    # Remove invalid rows (e.g., non-numeric pm2_5 values)
    #df = df[pd.to_numeric(df["pm2_5"], errors="coerce").notnull()]
    
    # Ensure numeric types for relevant columns
    numeric_cols = ["aqi", "temperature", "pressure", "humidity", "wind_speed"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    return df


if __name__ == "__main__":
    # Directory containing AirVisual JSON files
    data_dir = "data/"
    
    # Load and preprocess the data
    raw_data = load_airvisual_data(data_dir)
    processed_data = preprocess_airvisual_data(raw_data)
    print(raw_data.info())
    
    # Save the cleaned data to a CSV file
    processed_data.to_csv("processed_data.csv")
    print("Data preprocessing completed.")
