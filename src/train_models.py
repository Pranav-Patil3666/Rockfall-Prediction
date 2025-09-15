# src/train_models.py
"""
Training script for KNN, Perceptron and a DNN.
This version:
 - Uses SMOTE to balance the training set (so we DO NOT pass class_weight to Keras).
 - Sets random seeds for reproducibility.
 - Saves scaler and models to outputs/models/.
 - Provides clearer logging.
If you prefer using class_weight instead of SMOTE, set USE_SMOTE = False.
"""

import os
import random
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Config
# -------------------------------
DATA_PATH = "data/jharia_dataset_with_dem_fixed.csv"
MODEL_DIR = "outputs/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Toggle behavior:
USE_SMOTE = True  # If False, do not SMOTE and pass class_weight to DNN instead

# Seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# -------------------------------
# Helper functions
# -------------------------------
def load_dataset():
    """Load dataset and return features + labels."""
    df = pd.read_csv(DATA_PATH)
    # Basic column checks
    required = ["slope", "rainfall_mm", "displacement_mm_per_hr", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    X = df[["slope", "rainfall_mm", "displacement_mm_per_hr"]].copy()
    y = df["label"].astype(int).copy()  # ensure integer labels
    print(f"📊 Label distribution (full dataset): {Counter(y)}")
    return X, y


def build_scaler(X_train):
    """Fit and return a standard scaler."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_artifact(obj, name):
    """Save model or scaler using joblib (or model.save where applicable)."""
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(obj, path)
    print(f"💾 Saved {name} -> {path}")


# -------------------------------
# Main training function
# -------------------------------
def train_and_evaluate():
    # 1. Load data
    print("📂 Loading dataset...")
    X, y = load_dataset()

    # 2. Train-test split (keep stratify for class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. Scale features
    scaler = build_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_artifact(scaler, "scaler.pkl")

    # 4. Handle class imbalance
    if USE_SMOTE:
        print("⚖️ Using SMOTE to balance the training set...")
        smote = SMOTE(random_state=SEED)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
        print(f"⚖️ After SMOTE balancing: {Counter(y_train_bal)}")
        dnn_train_X, dnn_train_y = X_train_bal, y_train_bal
        dnn_class_weight = None
    else:
        # Do not SMOTE; compute class weights for Keras
        print("⚖️ Not using SMOTE; will compute class_weight for DNN.")
        X_train_bal, y_train_bal = X_train_scaled, y_train  # keep original training set
        # compute class weights
        class_counts = Counter(y_train_bal)
        total = sum(class_counts.values())
        class_weight = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        print(f"⚖️ Computed class_weight: {class_weight}")
        dnn_train_X, dnn_train_y = X_train_bal, y_train_bal
        dnn_class_weight = class_weight

    # ---------------------------
    # Model 1: KNN
    # ---------------------------
    print("🤖 Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bal, y_train_bal)
    knn_acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"✅ KNN Test Accuracy: {knn_acc:.4f}")
    save_artifact(knn, "knn_model.pkl")

    # ---------------------------
    # Model 2: Perceptron
    # ---------------------------
    print("🤖 Training Perceptron...")
    perceptron = Perceptron(max_iter=1000, random_state=SEED)
    perceptron.fit(X_train_bal, y_train_bal)
    perc_acc = accuracy_score(y_test, perceptron.predict(X_test_scaled))
    print(f"✅ Perceptron Test Accuracy: {perc_acc:.4f}")
    save_artifact(perceptron, "perceptron_model.pkl")

    # ---------------------------
    # Model 3: Deep Neural Network
    # ---------------------------
    print("🤖 Training DNN...")

    input_dim = dnn_train_X.shape[1]
    dnn = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    dnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    fit_kwargs = dict(
        x=dnn_train_X,
        y=dnn_train_y,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # Only pass class_weight if we are NOT using SMOTE (i.e., USE_SMOTE == False)
    if dnn_class_weight is not None:
        fit_kwargs["class_weight"] = dnn_class_weight
        print(" - Passing class_weight to model.fit (since SMOTE not used).")
    else:
        print(" - Not passing class_weight to model.fit (SMOTE used).")

    dnn.fit(**fit_kwargs)

    dnn_loss, dnn_acc = dnn.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"✅ DNN Test Accuracy: {dnn_acc:.4f}")

    # Save Keras model using native save
    dnn.save(os.path.join(MODEL_DIR, "dnn_model.h5"))
    print(f"💾 Saved dnn_model.h5 -> {MODEL_DIR}")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    print("🚀 Training script started...")
    train_and_evaluate()
    print("🎉 All models trained and saved.")
