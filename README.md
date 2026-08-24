# BuildCost AI — Future Building Construction Cost Prediction

AI-powered web application and API to estimate future residential building construction costs (e.g., Year 2027 and beyond) based on historical building data, structural parameters, material consumption, labour metrics, and inflation indices.

---

## 📁 Repository Structure

```text
├── backend/
│   ├── main.py                     # FastAPI REST server & prediction pipeline
│   └── models.py                   # Pydantic validation schemas & PyTorch Neural Network
├── frontend/
│   ├── index.html                  # Modern responsive dashboard
│   ├── styles.css                  # Custom styling & print layout
│   └── app.js                      # Client logic & live estimation
├── models/
│   ├── future_construction_cost_model.pkl   # Serialized preprocessor + XGBoost pipeline
│   └── pytorch_construction_model.pth       # PyTorch Deep Neural Network state weights
├── dataset/
│   └── Construction_ML_Dataset_1000.csv     # 1,000 historical project records (2010–2026)
├── run.py                          # 1-click Python launcher
├── start.bat                       # Windows double-click launcher
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation
```

---

## ⚡ Quick Start Guide

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run Application
```bash
python run.py
```
*(Or double-click `start.bat` on Windows)*

* 🌐 **Web Dashboard UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* 📚 **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧠 Accuracy & Methodology

* **Chronological Testing:** Trained on historical projects (2010–2024), validated on 2025, and tested on unseen 2026 data.
* **Leakage Prevention:** Strictly isolated `Material_Cost` and `Labour_Cost` to prevent artificial inflation or data leakage.
* **Validation Performance:** $R^2 = 85.79\%$, $\text{MAPE} = 12.38\%$.
