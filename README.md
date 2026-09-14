🏗️ BuildCost AI – Construction Cost Prediction

📌 Project Overview

BuildCost AI is an AI-powered construction cost prediction system designed to estimate future residential building construction costs. The system uses historical construction data, building parameters, material quantities, labour metrics, material rate information, and machine learning models to generate cost estimates.

The project combines a FastAPI backend, a web-based frontend, and trained XGBoost and PyTorch models to provide construction cost predictions through an interactive dashboard and REST API.

🚀 Features

🔮 Future residential construction cost prediction

🤖 Machine learning-based cost estimation

🌳 XGBoost model for the best-performing prediction

🧠 PyTorch deep neural network for prediction comparison

📊 Comparison with Random Forest, Linear Regression, and CatBoost

🏠 Supports multiple building and construction parameters

🧱 Material quantity and labour-based cost breakdown

📈 Construction cost trends and metadata

🌐 Interactive web dashboard

⚡ FastAPI REST API

📚 Swagger API documentation

🛡️ Data leakage prevention

📅 Chronological train, validation, and test strategy

🧠 Machine Learning Models

The system evaluates multiple machine learning approaches:

Model

R² Score

MAPE

MAE

XGBoost Tuned

0.8579

12.38%

₹19,92,493

Random Forest

0.8549

11.76%

₹19,38,307

Linear Regression

0.8474

13.28%

₹20,82,620

PyTorch Deep NN

0.8395

12.58%

₹20,58,362

CatBoost

0.8241

13.83%

₹21,67,312

🏆 Best Performing Model

The Tuned XGBoost model provides the best overall R² performance:

R²: 85.79%

MAPE: 12.38%

MAE: ₹19.92 Lakhs

RMSE: ₹25.04 Lakhs

🗂️ Dataset

The project uses:

Construction_ML_Dataset_1000.csv

Dataset Details

Records: 1,000 historical construction projects

Years: 2010–2026

Application: Residential construction cost prediction

Testing: Unseen 2026 construction data

The chronological evaluation strategy is:

Training Data  →  2010–2024
Validation     →  2025
Testing        →  Unseen 2026

This approach helps evaluate how well the model can predict future construction costs rather than simply memorizing randomly split historical records.

🛡️ Data Leakage Prevention

The project separates material and labour cost information to reduce artificial inflation and data leakage during model training.

The prediction pipeline primarily uses construction characteristics, quantities, rates, building information, and engineered features to estimate future costs.

⚙️ Feature Engineering

The backend generates additional features from the input data:

Building_Age

Area_per_Floor

Cement_per_sqft

Steel_per_sqft

Sand_per_sqft

Brick_per_sqft

Total_Labour_Days

Labour_Day_per_sqft

Material_Rate_Index

These derived features help the models capture relationships between building size, material consumption, labour requirements, and construction cost.

🏗️ System Architecture

                 ┌─────────────────────────┐
                 │      Web Frontend        │
                 │ HTML + CSS + JavaScript  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       FastAPI           │
                 │       Backend           │
                 └────────────┬────────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
       ┌───────────┐    ┌───────────┐    ┌───────────┐
       │ XGBoost   │    │  PyTorch  │    │ Historical│
       │   Model   │    │    NN     │    │   Data    │
       └─────┬─────┘    └─────┬─────┘    └───────────┘
             │                │
             └────────┬───────┘
                      ▼
             ┌──────────────────┐
             │ Cost Prediction  │
             │ + Cost Breakdown │
             └──────────────────┘

🔄 Prediction Workflow

User Input
    ↓
Building & Construction Parameters
    ↓
Feature Engineering
    ↓
Data Preprocessing
    ↓
Machine Learning Models
    ↓
XGBoost + PyTorch Prediction
    ↓
Cost Calculation & Breakdown
    ↓
Final Construction Cost Estimate

📁 Project Structure

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
│   ├── app.js
│   ├── index.html
│   └── styles.css
│
├── models/
│   ├── future_construction_cost_model.pkl
│   └── pytorch_construction_model.pth
│
├── README.md
├── requirements.txt
├── run.py
└── start.bat

🛠️ Technologies Used

Backend

Python

FastAPI

Uvicorn

Pydantic

Machine Learning

XGBoost

PyTorch

Scikit-learn

CatBoost

Random Forest

Linear Regression

Data Processing

Pandas

NumPy

Joblib

Frontend

HTML5

CSS3

JavaScript

📦 Python Libraries

Install the required dependencies using:

pip install -r requirements.txt

Main dependencies include:

fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
joblib
torch
xgboost
catboost

💻 Installation

1. Clone the Repository

git clone <your-repository-url>
cd construction-price-prediction-

2. Create a Virtual Environment

python -m venv venv

3. Activate the Virtual Environment

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

4. Install Dependencies

pip install -r requirements.txt

▶️ Running the Application

You can start the application using:

python run.py

Or on Windows:

start.bat

The FastAPI server runs at:

http://127.0.0.1:8000

🌐 Application

Open the following URL in your browser:

http://127.0.0.1:8000

The web interface allows users to enter building and construction information and receive a predicted construction cost.

📚 API Documentation

FastAPI automatically provides Swagger documentation.

Open:

http://127.0.0.1:8000/docs

Main API Endpoints

Endpoint

Method

Purpose

/api/health

GET

Check API health

/api/metadata

GET

Retrieve cities, qualities, and historical trends

/api/evaluation

GET

View model evaluation results

/api/predict

POST

Generate construction cost prediction

🧾 Prediction Inputs

The prediction API accepts building, material, labour, and rate-related information.

Building Parameters

Construction Year

City

Plot Area

Built-up Area

Number of Floors

House Type

Construction Quality

Bedroom Count

Bathroom Count

Hall Count

Kitchen Count

Foundation Type

Material Parameters

Steel Quantity

Cement Bags

Sand Quantity

Aggregate Quantity

Brick Quantity

Labour Parameters

Mason Labour Days

Carpenter Labour Days

Electrician Labour Days

Plumber Labour Days

Material Rates

Cement Rate per Bag

Steel Rate per kg

Sand Rate per m³

Brick Rate per 1000

📊 Model Evaluation

The model was evaluated using unseen construction data from 2026.

Evaluation Metrics

The project uses:

R² Score

MAPE

MAE

RMSE

Overall Performance

R² Score          : 85.79%
Mean Accuracy     : 87.62%
Average Error     : 12.38%
Test MAE          : ₹19.92 Lakhs
Test Year         : 2026
Training Years    : 2010–2024

🧠 PyTorch Deep Neural Network

The project also implements a PyTorch neural network for construction cost prediction.

Architecture:

Input Features
      ↓
Linear Layer
      ↓
Batch Normalization
      ↓
ReLU
      ↓
Dropout
      ↓
Linear Layer
      ↓
Batch Normalization
      ↓
ReLU
      ↓
Dropout
      ↓
Linear Layer
      ↓
ReLU
      ↓
Output Layer

The neural network is used alongside the XGBoost model for prediction comparison.

🎯 Project Objectives

Predict future residential construction costs.

Use historical construction data for machine learning.

Incorporate building, material, labour, and rate parameters.

Engineer meaningful construction-related features.

Compare different machine learning models.

Reduce data leakage during model development.

Provide predictions through a user-friendly web interface.

Expose the prediction system through a REST API.

🏠 Applications

This system can support:

Residential construction planning

Preliminary construction budgeting

Cost estimation

Contractor planning

Material planning

Labour planning

Construction project analysis

Early-stage financial decision making

📈 Key Machine Learning Concepts

Concept

Usage

Feature Engineering

Construction-specific derived features

Regression

Predict construction cost

XGBoost

Primary prediction model

Deep Learning

PyTorch neural network

Model Comparison

Evaluate multiple algorithms

Chronological Split

Future-oriented evaluation

Data Leakage Prevention

Avoid artificial prediction performance

Evaluation Metrics

R², MAPE, MAE, RMSE

🔍 Example Prediction Process

Input:
    Built-up Area
    Number of Floors
    Construction Quality
    Material Quantities
    Labour Days
    Material Rates
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
XGBoost Model
        +
PyTorch Model
        ↓
Predicted Construction Cost
        ↓
Cost per Square Foot
        +
Material/Labour Breakdown

🌟 Advantages

Uses real construction-related parameters.

Supports future cost prediction.

Uses multiple machine learning algorithms.

Provides model performance comparison.

Includes construction-specific feature engineering.

Provides an interactive web interface.

Provides REST API access.

Uses chronological evaluation to simulate future prediction.

Provides cost breakdown information.

⚠️ Limitations

Prediction accuracy depends on the quality and representativeness of the historical dataset.

Actual construction costs can vary based on market conditions, location, contractor pricing, material availability, and project-specific requirements.

The system is intended for estimation and planning rather than a final construction quotation.

🔮 Future Enhancements

Real-time material price integration

Live inflation and market index updates

Larger and more diverse construction datasets

Advanced time-series forecasting

Regional construction cost modeling

Cloud deployment

User authentication

Database integration

PDF cost estimation reports

Interactive cost trend visualization

Explainable AI for prediction interpretation

Mobile-friendly application

📌 Future Scope

The system can be extended into a complete AI-assisted construction planning platform by integrating:

Cost Prediction
      +
Material Estimation
      +
Labour Estimation
      +
Inflation Forecasting
      +
3D Building Visualization
      +
Project Budget Planning

⚠️ Disclaimer

The predicted construction cost is an estimated value generated using machine learning models and historical data. It should be used for preliminary planning and budgeting purposes only and should not be considered a final construction quotation.

📄 License

This project is intended for academic and educational purposes.

👨‍💻 Author

Your Name

Computer Science / Information Technology Student

⭐ Conclusion

BuildCost AI demonstrates how machine learning can be applied to residential construction cost estimation. By combining construction parameters, material quantities, labour metrics, engineered features, and multiple machine learning models, the system provides a practical approach for estimating future construction costs through both a web interface and REST API.

The Tuned XGBoost model achieved an R² score of 85.79% on the project's evaluation setup, making it the best-performing model among the evaluated approaches.
