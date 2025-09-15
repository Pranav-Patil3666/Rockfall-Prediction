# src/serve.py

import os
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, send_file

# -------------------------------
# Paths
# -------------------------------
MODEL_DIR = "outputs/models/"
REPORT_DIR = "outputs/reports/"
GEOJSON_PATH = os.path.join(REPORT_DIR, "predictions.geojson")

# -------------------------------
# Load Artifacts
# -------------------------------
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
knn = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
perceptron = joblib.load(os.path.join(MODEL_DIR, "perceptron_model.pkl"))
dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dnn_model.h5"))

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

@app.route("/predict_json", methods=["POST"])
def predict_json():
    """
    Input JSON example:
    {
        "slope": 35,
        "rainfall_mm": 100,
        "displacement_mm_per_hr": 0.08,
        "latitude": 23.7401,
        "longitude": 86.4145
    }
    """
    data = request.get_json()

    # Extract features
    X = pd.DataFrame([data])[["slope", "rainfall_mm", "displacement_mm_per_hr"]]
    X_scaled = scaler.transform(X)

    # Predictions
    knn_pred = int(knn.predict(X_scaled)[0])
    perc_pred = int(perceptron.predict(X_scaled)[0])
    dnn_prob = float(dnn.predict(X_scaled, verbose=0).flatten()[0])
    dnn_pred = int(dnn_prob > 0.5)

    # Risk flag
    risk_level = "HIGH" if dnn_prob > 0.5 else "LOW"

    return jsonify({
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "knn_pred": knn_pred,
        "perceptron_pred": perc_pred,
        "dnn_prob": round(dnn_prob, 3),
        "dnn_pred": dnn_pred,
        "risk_level": risk_level
    })

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """
    Upload CSV file with columns:
    slope, rainfall_mm, displacement_mm_per_hr, latitude, longitude
    """
    if "file" not in request.files:
        return "❌ No file uploaded", 400

    file = request.files["file"]
    df = pd.read_csv(file)

    X = df[["slope", "rainfall_mm", "displacement_mm_per_hr"]]
    X_scaled = scaler.transform(X)

    df["knn_pred"] = knn.predict(X_scaled)
    df["perceptron_pred"] = perceptron.predict(X_scaled)
    df["dnn_prob"] = dnn.predict(X_scaled, verbose=0).flatten()
    df["dnn_pred"] = (df["dnn_prob"] > 0.5).astype(int)
    df["risk_level"] = df["dnn_prob"].apply(lambda p: "HIGH" if p > 0.5 else "LOW")

    # Return JSON records (keeps lat/long if included)
    cols = ["latitude", "longitude", "slope", "rainfall_mm", "displacement_mm_per_hr",
            "knn_pred", "perceptron_pred", "dnn_prob", "dnn_pred", "risk_level"]
    df = df[[c for c in cols if c in df.columns]]

    return df.to_json(orient="records")

@app.route("/predict_bulk", methods=["GET"])
def predict_bulk():
    """
    Serves the latest predictions as GeoJSON file.
    Generated earlier by infer.py
    """
    if not os.path.exists(GEOJSON_PATH):
        return jsonify({"error": "GeoJSON not found. Run infer.py first."}), 404
    
    return send_file(GEOJSON_PATH, mimetype="application/geo+json")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask API server on http://127.0.0.1:5000")
    app.run(debug=True)
