from prometheus_client import start_http_server, Gauge
import requests
import time
import random

#METRICS DEFINITIONS 
temperature = Gauge('weather_temperature_celsius', 'Current temperature in Celsius', ['city'])
humidity = Gauge('weather_humidity_percent', 'Current humidity percentage', ['city'])
pressure = Gauge('weather_pressure_hpa', 'Atmospheric pressure in hPa', ['city'])
wind_speed = Gauge('weather_wind_speed_mps', 'Wind speed in meters per second', ['city'])
cloudiness = Gauge('weather_cloudiness_percent', 'Cloudiness percentage', ['city'])
visibility = Gauge('weather_visibility_meters', 'Visibility distance in meters', ['city'])
sunrise = Gauge('weather_sunrise_timestamp', 'Sunrise time (UNIX timestamp)', ['city'])
sunset = Gauge('weather_sunset_timestamp', 'Sunset time (UNIX timestamp)', ['city'])
temperature_feels = Gauge('weather_temperature_feels_like', 'Feels like temperature in Celsius', ['city'])
rain_volume = Gauge('weather_rain_volume_mm', 'Rain volume for the last hour (mm)', ['city'])

#CONFIGURATION
API_KEY = "f6f2999b8bc77c260041ccf5875b17ba"
CITIES = ["London", "New York", "Tokyo", "Berlin", "Paris"]

def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        r = requests.get(url)
        data = r.json()
        return data
    except Exception as e:
        print(f"Error fetching data for {city}: {e}")
        return None

def update_metrics():
    """Fetch data and update Prometheus metrics"""
    for city in CITIES:
        data = get_weather(city)
        if not data or data.get("cod") != 200:
            continue

        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})

        temperature.labels(city=city).set(main.get("temp", 0))
        temperature_feels.labels(city=city).set(main.get("feels_like", 0))
        humidity.labels(city=city).set(main.get("humidity", 0))
        pressure.labels(city=city).set(main.get("pressure", 0))
        wind_speed.labels(city=city).set(wind.get("speed", 0))
        cloudiness.labels(city=city).set(clouds.get("all", 0))
        visibility.labels(city=city).set(data.get("visibility", 0))
        sunrise.labels(city=city).set(data["sys"].get("sunrise", 0))
        sunset.labels(city=city).set(data["sys"].get("sunset", 0))
        rain_volume.labels(city=city).set(rain.get("1h", 0))

if __name__ == "__main__":
    start_http_server(9105)  
    print(" Custom Exporter running on http://localhost:9105/metrics")

    while True:
        update_metrics()
        time.sleep(20)  #20 seconds
