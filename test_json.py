
import requests

# Flask API endpoint
url = "http://127.0.0.1:5000/predict_json"

# Example input data
data = {
    "slope": 35,
    "rainfall_mm": 120,
    "displacement_mm_per_hr": 0.05,
    "latitude": 23.74,
    "longitude": 86.41
}

try:
    response = requests.post(url, json=data)
    print("✅ Status Code:", response.status_code)
    print("✅ Response JSON:", response.json())
except Exception as e:
    print("❌ Error:", e)
