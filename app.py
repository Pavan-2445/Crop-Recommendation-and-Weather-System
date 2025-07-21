import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
import joblib
import numpy as np
import requests


load_dotenv()  # Load environment variables from .env

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder (1).pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {
        'nitrogen': '',
        'phosphorus': '',
        'potassium': '',
        'ph': '',
        'temperature': '',
        'humidity': '',
        'rainfall': ''
    }
    if request.method == "POST":
        try:
            # Extract form data
            N = request.form["nitrogen"]
            P = request.form["phosphorus"]
            K = request.form["potassium"]
            ph = request.form["ph"]
            temperature = request.form["temperature"]
            humidity = request.form["humidity"]
            rainfall = request.form["rainfall"]
            # Save for repopulation
            form_data = {
                'nitrogen': N,
                'phosphorus': P,
                'potassium': K,
                'ph': ph,
                'temperature': temperature,
                'humidity': humidity,
                'rainfall': rainfall
            }
            # Prepare data for prediction
            input_data = np.array([[float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]])
            pred_encoded = model.predict(input_data)[0]
            prediction = label_encoder.inverse_transform([pred_encoded])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction, active_tab='crop', form_data=form_data)

@app.route("/weather", methods=["GET", "POST"])
def weather():
    weather = None
    weather_error = None
    form_data = {
        'nitrogen': '',
        'phosphorus': '',
        'potassium': '',
        'ph': '',
        'temperature': '',
        'humidity': '',
        'rainfall': ''
    }
    if request.method == "POST":
        city = request.form.get("city")
        try:
            # 1. Use Nominatim to get lat/lon
            nominatim_url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': city,
                'format': 'json',
                'limit': 1
            }
            nom_response = requests.get(nominatim_url, params=params, headers={"User-Agent": "Mozilla/5.0"})
            nom_data = nom_response.json()
            if not nom_data:
                raise Exception("Location not found.")
            lat = nom_data[0]['lat']
            lon = nom_data[0]['lon']

            # 2. Use WeatherAPI.com to get weather
            WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
            if not WEATHER_API_KEY:
                raise Exception("Weather API key not set in environment.")
            weather_url = f"http://api.weatherapi.com/v1/current.json"
            weather_params = {
                'key': WEATHER_API_KEY,
                'q': f"{lat},{lon}",
                'aqi': 'no'
            }
            w_response = requests.get(weather_url, params=weather_params)
            w_data = w_response.json()
            if w_response.status_code != 200 or 'current' not in w_data:
                raise Exception(w_data.get('error', {}).get('message', 'Weather API error.'))
            weather = {
                'city': w_data['location']['name'],
                'temp': w_data['current']['temp_c'],
                'humidity': w_data['current']['humidity'],
                'description': w_data['current']['condition']['text']
            }
        except Exception as e:
            weather_error = str(e)
    return render_template("index.html", weather=weather, weather_error=weather_error, active_tab='weather', form_data=form_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
