# retrain/retrain.py

import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import tensorflow as tf

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/jharia_dataset_with_dem.csv"
MODEL_DIR = "outputs/models/"
LOG_DIR = "outputs/logs/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------
# Main retrain function
# -------------------------------
def retrain():
    print("📂 Loading dataset for retraining...")
    df = pd.read_csv(DATA_PATH)
    X = df[["slope", "rainfall_mm", "displacement_mm_per_hr"]]
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Versioning timestamp
    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save scaler (versioned + latest copy)
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{version}.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))  # overwrite latest

    # ---------------------------
    # Model 1: KNN
    # ---------------------------
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    joblib.dump(knn, os.path.join(MODEL_DIR, f"knn_model_{version}.pkl"))
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn_model.pkl"))  # overwrite latest

    # ---------------------------
    # Model 2: Perceptron
    # ---------------------------
    perceptron = Perceptron(max_iter=1000, random_state=42)
    perceptron.fit(X_train_scaled, y_train)
    perc_acc = accuracy_score(y_test, perceptron.predict(X_test_scaled))
    joblib.dump(perceptron, os.path.join(MODEL_DIR, f"perceptron_model_{version}.pkl"))
    joblib.dump(perceptron, os.path.join(MODEL_DIR, "perceptron_model.pkl"))  # overwrite latest

    # ---------------------------
    # Model 3: Deep Neural Network
    # ---------------------------
    dnn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    dnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    dnn.fit(X_train_scaled, y_train, epochs=20, batch_size=8, verbose=0)

    dnn_loss, dnn_acc = dnn.evaluate(X_test_scaled, y_test, verbose=0)
    dnn.save(os.path.join(MODEL_DIR, f"dnn_model_{version}.h5"))
    dnn.save(os.path.join(MODEL_DIR, "dnn_model.h5"))  # overwrite latest

    # ---------------------------
    # Logging retrain run
    # ---------------------------
    log_path = os.path.join(LOG_DIR, "retrain_log.txt")
    with open(log_path, "a") as f:
        f.write(f"[{version}] KNN={knn_acc:.2f}, Perceptron={perc_acc:.2f}, DNN={dnn_acc:.2f}\n")

    print(f"✅ Retraining complete. Models saved with version {version}.")
    print(f"📊 Accuracies -> KNN: {knn_acc:.2f}, Perceptron: {perc_acc:.2f}, DNN: {dnn_acc:.2f}")
    print(f"📝 Log updated at {log_path}")

    # ---------------------------
    # Auto-run inference after retraining
    # ---------------------------
    print("🔄 Running inference to regenerate predictions (CSV + GeoJSON)...")
    os.system("python src/infer.py")


if __name__ == "__main__":
    retrain()
