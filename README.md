**Rockfall Prediction using Machine Learning & Geospatial Analysis**

This project focuses on predicting rockfall hazards by combining geospatial terrain analysis with machine learning models. It leverages Digital Elevation Models (DEMs), rainfall data, and ground displacement rates to assess slope stability and classify risk levels (HIGH / LOW).

**🚀 Key Features**

**DEM Preprocessing**

Reprojects DEM into appropriate UTM zones for accurate slope computation in meters.

Computes slope maps (degrees) using gradient-based methods.

Attaches slope values to location points (lat/lon) in the dataset.



**Machine Learning Models**

Implements KNN Classifier, Perceptron, and a Deep Neural Network (DNN).

Trained with SMOTE to handle class imbalance for reliable predictions.

Standardized features using Scikit-learn’s StandardScaler.


**API Service (Flask)**

REST endpoints to get predictions via JSON or CSV.

Provides real-time risk classification (HIGH or LOW) based on slope, rainfall, and displacement inputs.

Supports bulk predictions exported as GeoJSON for visualization in GIS tools.

**End-to-End Workflow**

1. 00_prep.py → Preprocess DEM & dataset (compute slope, clean data).

2. train_models.py → Train ML models (KNN, Perceptron, DNN) & save artifacts.

3. serve.py → Launch Flask server for interactive predictions.

4. infer.py (optional) → Run bulk inference and generate reports.

5. evaluate.py → Model evaluation & performance metrics.


**📊 Input Features**

1. Slope (degrees) – derived from DEM.

2. Rainfall (mm) – precipitation data.

3. Displacement (mm/hr) – ground movement rate.


**🔎 Output**

1. Risk Level: HIGH or LOW.

2. Model predictions: KNN, Perceptron, DNN with probabilities.

3. GeoJSON outputs: Mapping predicted hazards.


**🛠️ Tech Stack**

Python, NumPy, Pandas, Scikit-learn, TensorFlow/Keras

Rasterio, PyProj (geospatial preprocessing)

Flask (API service)

SMOTE (imbalanced data handling)
