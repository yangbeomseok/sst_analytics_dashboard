# 🌊 SST-Analytics Dashboard

> **Sea Surface Temperature (SST) Prediction Application** > A project that predicts the sea surface temperature of the Busan coast using a LightGBM-based machine learning model and provides intuitive verification through a Streamlit web application.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Project Overview

Sea Surface Temperature (SST) in the coastal area of Busan plays a crucial role in various fields, including **climate change research, marine ecosystem management, and fishing activities**.  
This project trains a **LightGBM model** using **ERA5 reanalysis data** to predict SST and visualizes the results with a **Streamlit application** to make it easily accessible to everyone.

---

## ✨ Key Features

- **Real-time SST Prediction**: Outputs SST prediction results for a specific date/condition input.
- **Time Series Analysis**: Estimates future SST changes based on past data.
- **Visualization**: Compares actual vs. predicted values in a graph.
- **User Input Support**: Allows custom predictions through CSV uploads.
- **Model Explainability**: Provides feature importance.

---

## 🛠 Tech Stack

- **Language**: Python 3.8+
- **Model**: LightGBM
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Others**: joblib/pickle (for model saving and loading)

---

## 📊 Data Description

- **Source**: ECMWF ERA5 Reanalysis Data
- **Data Format**: CSV (train, validation, test)
- **Main Features**:
  - Temperature
  - Sea Level Pressure
  - Wind Speed
  - Precipitation
  - Humidity
  - Other climate/ocean variables
- **Target**: Sea Surface Temperature (SST, ℃)

---

## ⚙️ Model Architecture & Training

- **Algorithm**: LightGBM (Gradient Boosting Decision Tree)
- **Hyperparameters (Example)**:
  ```yaml
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  n_estimators: 500
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  
- **Performance Evaluation Metrics**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)

---

## 📂 Project Structure

SST_Prediction_APP/  
├── LightGBM_modeling.py      # LightGBM 모델 학습 및 평가  
├── LightGBM_app.py           # Streamlit 앱 실행 스크립트  
├── requirements.txt          # 필요 라이브러리  
├── data/  
│   ├── train_data.csv  
│   ├── validation_data.csv  
│   └── test_data.csv  
└── models/  
    └── sst_model.pkl         # 학습 완료된 모델

---

## 📈 Usage Examples

- **Single Input Prediction**
  - Input date, temperature, wind speed, pressure, etc., to get a predicted SST value.

- **CSV Upload Prediction**
  - Upload a historical data file → Predict SST for multiple time points.

- **Visualization Results**
  - Actual vs. Predicted values comparison graph.
  - Feature Importance plot.

---

## 🔮 Future Improvements

- Comparative experiments with deep learning models (LSTM, Transformer).
- Spatial prediction (map-based SST prediction).
- Automated hyperparameter optimization using tools like Optuna.
- Deployment via Docker and integration with cloud services.
- Providing a REST API for integration with external services.
