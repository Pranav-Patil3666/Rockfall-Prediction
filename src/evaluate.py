# src/evaluate.py

import os
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/jharia_dataset_with_dem_fixed.csv"
MODEL_DIR = "outputs/models/"

# -------------------------------
# Load Artifacts
# -------------------------------
print("📦 Loading trained models and scaler...")
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
knn = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
perceptron = joblib.load(os.path.join(MODEL_DIR, "perceptron_model.pkl"))
dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dnn_model.h5"))

# -------------------------------
# Load dataset
# -------------------------------
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
X = df[["slope", "rainfall_mm", "displacement_mm_per_hr"]]
y = df["label"]

# Train-test split (same as training script)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Evaluate Models
# -------------------------------
def evaluate_model(name, model, X, y_true, is_dnn=False):
    print(f"\n🔎 Evaluating {name}...")
    if is_dnn:
        y_prob = model.predict(X, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)

    print(f"✅ Accuracy: {acc:.3f}")
    print("📊 Confusion Matrix:\n", cm)
    print("📝 Classification Report:\n", report)


# Run evaluations
evaluate_model("KNN", knn, X_test_scaled, y_test)
evaluate_model("Perceptron", perceptron, X_test_scaled, y_test)
evaluate_model("DNN", dnn, X_test_scaled, y_test, is_dnn=True)
