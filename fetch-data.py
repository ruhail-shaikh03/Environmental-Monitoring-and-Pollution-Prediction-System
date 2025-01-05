import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# API endpoints and keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AIRVISUAL_API_KEY = os.getenv("AIRVISUAL_API_KEY")

def fetch_openweather_data():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=40.7128&lon=-74.0060&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_airvisual_data():
    url = f"https://api.airvisual.com/v2/nearest_city?key={AIRVISUAL_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def save_data(data, filename):
    os.makedirs("data", exist_ok=True)
    filepath = f"data/{filename}"
    with open(filepath, "w") as f:
        f.write(data)
    print(f"Data saved to {filepath}")
    return filepath



if __name__ == "__main__":
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Fetch and save data from OpenWeather
        openweather_data = fetch_openweather_data()
        #save_data(str(openweather_data), f"openweather_{timestamp}.json")
        
        # Fetch and save data from AirVisual
        airvisual_data = fetch_airvisual_data()
        save_data(str(airvisual_data), f"airvisual_{timestamp}.json")

    except Exception as e:
        print(f"Error fetching data: {e}")
