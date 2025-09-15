# src/infer.py
"""
Inference script:
 - Loads trained scaler + models
 - Runs predictions on full dataset
 - Uses consistent 0.5 threshold for classification and risk levels
 - Saves predictions to CSV and GeoJSON
"""

import os
import pandas as pd
import joblib
import tensorflow as tf
from shapely.geometry import Point
import geopandas as gpd

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/jharia_dataset_with_dem_fixed.csv"   # DEM-based dataset
MODEL_DIR = "outputs/models/"
REPORT_DIR = "outputs/reports/"
CSV_PATH = os.path.join(REPORT_DIR, "predictions.csv")
GEOJSON_PATH = os.path.join(REPORT_DIR, "predictions.geojson")
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------------------
# Load Artifacts
# -------------------------------
print("📦 Loading trained models and scaler...")
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
knn = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
perceptron = joblib.load(os.path.join(MODEL_DIR, "perceptron_model.pkl"))
dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dnn_model.h5"))

# -------------------------------
# Inference
# -------------------------------
print("📂 Loading dataset for inference...")
df = pd.read_csv(DATA_PATH)

X = df[["slope", "rainfall_mm", "displacement_mm_per_hr"]]
X_scaled = scaler.transform(X)

print("🤖 Running predictions...")
df["knn_pred"] = knn.predict(X_scaled)
df["perceptron_pred"] = perceptron.predict(X_scaled)
df["dnn_prob"] = dnn.predict(X_scaled, verbose=0).flatten()

# Consistent 0.5 threshold
df["dnn_pred"] = (df["dnn_prob"] >= 0.5).astype(int)
df["risk_level"] = df["dnn_pred"].map({1: "HIGH", 0: "LOW"})

# -------------------------------
# Save Results
# -------------------------------
# Save as CSV
df.to_csv(CSV_PATH, index=False)
print(f"✅ Predictions CSV updated -> {CSV_PATH}")

# Save as GeoJSON (for frontend map)
if "latitude" in df.columns and "longitude" in df.columns:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)],
        crs="EPSG:4326"
    )
    gdf.to_file(GEOJSON_PATH, driver="GeoJSON")
    print(f"🗺️ Predictions GeoJSON updated -> {GEOJSON_PATH}")
else:
    print("⚠️ No latitude/longitude columns found — GeoJSON not generated.")
