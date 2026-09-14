# 🏗️ BuildCost AI – Construction Cost Prediction

## 📌 Overview

BuildCost AI is a machine learning-based system that predicts future residential construction costs using historical construction data, building parameters, material quantities, labour details, and material rates.

The project uses **XGBoost and PyTorch** models with a **FastAPI backend** and a web-based frontend.

## 🚀 Features

* Future construction cost prediction
* XGBoost and PyTorch models
* Material and labour cost estimation
* Construction cost per square foot
* Model performance comparison
* Web-based prediction interface
* REST API using FastAPI
* Swagger API documentation

## 🧠 Machine Learning Models

The project compares:

* XGBoost
* Random Forest
* Linear Regression
* PyTorch Neural Network
* CatBoost

### 🏆 Best Result

**XGBoost**

* R² Score: **85.79%**
* MAPE: **12.38%**
* MAE: **₹19.92 Lakhs**

## 📊 Dataset

* **Dataset:** `Construction_ML_Dataset_1000.csv`
* **Records:** 1,000
* **Years:** 2010–2026
* **Training:** 2010–2024
* **Validation:** 2025
* **Testing:** 2026

## ⚙️ Technologies Used

* Python
* FastAPI
* XGBoost
* PyTorch
* Scikit-learn
* CatBoost
* Pandas
* NumPy
* HTML
* CSS
* JavaScript

## 📁 Project Structure

```text
construction-price-prediction-/
│
├── backend/
│   ├── main.py
│   └── models.py
│
├── dataset/
│   └── Construction_ML_Dataset_1000.csv
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── models/
│   ├── future_construction_cost_model.pkl
│   └── pytorch_construction_model.pth
│
├── requirements.txt
├── run.py
└── README.md
```

## 💻 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd construction-price-prediction-
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Start the application using:

```bash
python run.py
```

The application will be available at:

`http://127.0.0.1:8000`

## 📚 API Documentation

FastAPI Swagger documentation:

`http://127.0.0.1:8000/docs`

### Main API Endpoints

| Endpoint          | Method | Description               |
| ----------------- | ------ | ------------------------- |
| `/api/health`     | GET    | Check API status          |
| `/api/metadata`   | GET    | Get project metadata      |
| `/api/evaluation` | GET    | Get model evaluation      |
| `/api/predict`    | POST   | Predict construction cost |

## 🔄 Prediction Process

```text
User Input
    ↓
Feature Engineering
    ↓
Data Preprocessing
    ↓
Machine Learning Model
    ↓
Cost Prediction
    ↓
Cost Breakdown
```

## 🎯 Project Objective

The main objective is to develop an AI-based system that provides estimated future construction costs to support preliminary budgeting and construction planning.

## 🔮 Future Enhancements

* Real-time material prices
* Inflation forecasting
* Larger datasets
* Cloud deployment
* PDF cost reports
* Advanced cost visualization
* Database integration

## ⚠️ Disclaimer

The predicted cost is an estimated value generated using machine learning and historical data. It should be used for preliminary planning and budgeting and not as a final construction quotation.

## ⭐ Conclusion

BuildCost AI demonstrates the use of machine learning for future residential construction cost prediction by combining construction parameters, material quantities, labour details, and material rates.
