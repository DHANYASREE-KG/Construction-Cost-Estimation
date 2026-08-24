# 🏗️ BuildCost AI — Future Building Construction Cost Prediction System

An end-to-end Machine Learning and PyTorch Deep Learning web application designed to forecast future building construction costs (e.g., Year 2027) based on architectural specifications, material quantities, labour days, and economic rate indices.

---

## 📁 Handover Project Structure

```text
Construction_Cost_Prediction_Project/
├── backend/
│   ├── main.py                    # FastAPI server & REST API routes
│   └── models.py                  # Pydantic schemas & PyTorch DNN architecture
├── frontend/
│   ├── index.html                 # Modern glassmorphism web UI
│   ├── styles.css                 # Responsive styling & themes
│   └── app.js                     # Live cost calculator & Chart.js visualizations
├── models/
│   ├── future_construction_cost_model.pkl  # Trained ML pipeline (XGBoost + Preprocessor)
│   └── pytorch_construction_model.pth      # Trained PyTorch Neural Network weights
├── dataset/
│   └── Construction_ML_Dataset_1000.csv    # Historical construction project records
├── run.py                         # 1-Click Python server launcher
├── start.bat                      # Windows double-click launcher
├── requirements.txt               # Required Python packages
└── README.md                      # Documentation
```

---

## ⚡ Quick Start Guide

### Step 1: Install Dependencies
Open your terminal in this project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Launch Application
Run the launcher script:
```bash
python run.py
```
*(Or double-click `start.bat` on Windows)*

* 🌐 **Web Dashboard UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* 📚 **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧠 Model Architecture & Methodology

1. **Data Leakage Elimination:**
   - Excludes `Material_Cost` and `Labour_Cost` because their direct sum is the target `Total_Estimated_Cost`.
   - The model learns from fundamental physical quantities (Cement bags, Steel kg, Sand m³, Brick count), labour days, and market unit rates.
2. **Chronological Splitting:**
   - Trains on historical years (2010–2024), validates on 2025, and tests on unseen 2026 data to prevent temporal lookahead leakage.
3. **Dual Model Engine:**
   - **XGBoost Regressor ($R^2 = 0.8579$):** Primary tabular gradient boosting engine.
   - **PyTorch Deep Neural Network (DNN):** 4-layer neural network with Batch Normalization, ReLU activations, and Dropout regularization.

---

## 🔌 API Reference

### 1. Predict Future Cost
* **Endpoint:** `POST /api/predict`
* **Content-Type:** `application/json`
* **Sample Request Body:**
```json
{
  "Construction_Year": 2027,
  "City": "Chennai",
  "Plot_Area_sqft": 2400.0,
  "Builtup_Area_sqft": 2000.0,
  "Number_of_Floors": 2,
  "House_Type": "Residential",
  "Construction_Quality": "Standard",
  "Bedroom_Count": 3,
  "Bathroom_Count": 3,
  "Hall_Count": 1,
  "Kitchen_Count": 1,
  "Foundation_Type": "RCC",
  "Steel_Quantity_kg": 9000.0,
  "Cement_Bags": 660.0,
  "Sand_Quantity_m3": 30.0,
  "Aggregate_Quantity_m3": 24.0,
  "Brick_Quantity": 16000.0,
  "Electrical_Points": 35,
  "Plumbing_Points": 22,
  "Mason_Labour_Days": 80.0,
  "Carpenter_Labour_Days": 50.0,
  "Electrician_Labour_Days": 25.0,
  "Plumber_Labour_Days": 22.0,
  "Cement_Rate_per_Bag": 420.0,
  "Steel_Rate_per_kg": 82.0,
  "Sand_Rate_per_m3": 2200.0,
  "Brick_Rate_per_1000": 11000.0
}
```

* **Sample Response:**
```json
{
  "status": "success",
  "target_year": 2027,
  "predicted_cost_xgb": 15781414.0,
  "predicted_cost_pytorch": 14093814.09,
  "best_estimate_inr": "₹15,781,414.00",
  "best_estimate_lakhs": "₹157.81 Lakhs",
  "cost_per_sqft": "₹7,890.71 / sq.ft",
  "breakdown": {
    "materials_direct": 10243200.0,
    "labour_direct": 212400.0,
    "equipment_and_overheads": 5325814.0
  },
  "selected_model": "XGBoost Regressor + PyTorch DNN"
}
```

### 2. System Health
* **Endpoint:** `GET /api/health`
* **Response:**
```json
{
  "status": "online",
  "models": {
    "xgboost": true,
    "pytorch_dnn": true,
    "preprocessor": true
  }
}
```

---

## 👥 Handover Notes
* The `models/` directory contains pre-trained, production-ready weights. No retraining is required to run the web app.
* To customize port or host, edit `run.py` or execute:
  ```bash
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
  ```
