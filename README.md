# 🏗️ BuildCost AI – Construction Cost Prediction

## 📌 Project Overview

BuildCost AI is an AI-powered application designed to predict residential building construction costs using historical construction data and project parameters. The system combines machine learning and deep learning techniques with a FastAPI backend and web interface to provide data-driven preliminary construction cost estimates.

---

## 🚀 Features

- 🤖 AI-based construction cost prediction
- 🏠 Residential construction cost estimation
- 📊 Historical construction data analysis
- 📈 Future construction cost prediction
- 🧱 Construction material analysis
- 👷 Labour requirement analysis
- 💰 Preliminary construction cost estimation
- 📅 Year-based cost prediction
- 🧠 XGBoost machine learning model
- 🔥 PyTorch neural network
- ⚡ FastAPI backend
- 🌐 Interactive web interface
- 📚 Swagger API documentation
- 🔄 Automated feature engineering
- 📊 Model evaluation using regression metrics

---

## 🧠 Machine Learning Models

### XGBoost

The project uses an **XGBoost regression model** for construction cost prediction. The model learns relationships between construction parameters and historical construction costs.

**Trained Model:**

```text
models/future_construction_cost_model.pkl
PyTorch

A PyTorch neural network is also implemented for construction cost prediction.

Trained Model:

models/pytorch_construction_model.pth
📊 Dataset

The project uses historical construction data containing information about residential construction projects.

The dataset includes parameters such as:

Construction year
City
Plot area
Built-up area
Number of floors
House type
Construction quality
Number of bedrooms
Number of bathrooms
Foundation type
Steel quantity
Cement quantity
Sand quantity
Aggregate quantity
Brick quantity
Electrical points
Plumbing points
Labour requirements
Material rates

Dataset:

dataset/Construction_ML_Dataset_1000.csv
⚙️ Feature Engineering

The system generates additional features from the original construction parameters to improve model performance.

Engineered Features
Building_Age
Area_per_Floor
Cement_per_sqft
Steel_per_sqft
Sand_per_sqft
Brick_per_sqft

These features help the machine learning models understand relationships between:

Building age
Building size
Floor area
Material consumption
Construction requirements
Material usage per square foot
🏗️ System Architecture
                         ┌───────────────────┐
                         │       User        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Web Interface   │
                         │    HTML/CSS/JS    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  FastAPI Backend  │
                         │      REST API     │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
           ┌─────────────────┐           ┌─────────────────┐
           │  XGBoost Model  │           │ PyTorch Model   │
           │   Regression    │           │ Neural Network  │
           └────────┬────────┘           └────────┬────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Construction Cost │
                         │    Prediction     │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  Display Result   │
                         └───────────────────┘
🔄 Prediction Workflow
User Input
    ↓
Input Validation
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Feature Transformation
    ↓
Machine Learning Model
    ↓
Construction Cost Prediction
    ↓
Result Processing
    ↓
Display Estimated Cost
📂 Project Structure
construction-price-prediction/
│
├── backend/
│   ├── main.py
│   └── models.py
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── models/
│   ├── future_construction_cost_model.pkl
│   └── pytorch_construction_model.pth
│
├── dataset/
│   └── Construction_ML_Dataset_1000.csv
│
├── run.py
├── start.bat
├── requirements.txt
└── README.md
🛠️ Technologies Used
Programming Language
Python 3
Machine Learning
XGBoost
PyTorch
Scikit-learn
Pandas
NumPy
Joblib
Backend
FastAPI
Pydantic
Uvicorn
Frontend
HTML
CSS
JavaScript
Data
CSV
Historical construction datasets
📦 Python Libraries

The main libraries used in the project include:

pandas
numpy
scikit-learn
xgboost
torch
fastapi
uvicorn
pydantic
joblib

Install all dependencies using:

pip install -r requirements.txt
⚙️ Installation
Step 1: Clone the Repository
git clone <repository-url>
Step 2: Navigate to the Project
cd construction-price-prediction
Step 3: Create a Virtual Environment
python -m venv venv
Step 4: Activate the Virtual Environment
Windows
venv\Scripts\activate
Linux / macOS
source venv/bin/activate
Step 5: Install Dependencies
pip install -r requirements.txt
▶️ Running the Application

Run the application using:

python run.py

For Windows, the application can also be started using:

start.bat
🌐 Access the Application

After starting the application, open:

http://127.0.0.1:8000

The web interface allows users to enter construction-related parameters and receive a predicted construction cost.

📚 API Documentation

The project uses FastAPI, which automatically provides interactive API documentation.

Open:

http://127.0.0.1:8000/docs

The Swagger UI can be used to:

View available API endpoints
Enter prediction parameters
Send requests
View API responses
Test the prediction service
🎯 Input Parameters

The prediction system can use construction parameters including:

Construction year
City
Plot area
Built-up area
Number of floors
House type
Construction quality
Number of bedrooms
Number of bathrooms
Hall and kitchen information
Foundation type
Steel quantity
Cement quantity
Sand quantity
Aggregate quantity
Brick quantity
Electrical points
Plumbing points
Labour requirements
Material rates
📈 Model Evaluation

The project evaluates construction cost prediction using historical construction data and future-oriented validation.

The model evaluation considers metrics such as:

R² Score
Mean Absolute Percentage Error (MAPE)
Prediction Accuracy

A chronological evaluation strategy can be used to simulate real-world future prediction.

Historical Data
      ↓
Training Data
      ↓
Validation Data
      ↓
Future/Test Data
      ↓
Model Evaluation
🔐 Data Leakage Prevention

The project considers data leakage during model development.

Target-related cost information is separated from the prediction features where appropriate so that the model does not simply reproduce the target value.

This allows the model to learn meaningful relationships from construction characteristics such as:

Building area
Construction year
Number of floors
Material consumption
Labour requirements
Building characteristics
🎯 Project Objectives

The main objectives of this project are:

Automate preliminary construction cost estimation
Apply machine learning to construction planning
Apply deep learning to construction cost prediction
Predict future construction costs
Reduce manual estimation effort
Analyze construction-related parameters
Provide data-driven cost estimates
Develop an easy-to-use web application
Demonstrate AI applications in residential construction
💡 Applications

The system can support:

🏠 Homeowners
🏗️ Builders
👷 Contractors
📐 Architects
📊 Quantity Surveyors
🏢 Real Estate Professionals
🏘️ Construction Planners

The system can provide an initial cost estimate during the planning stage before detailed professional estimation.

🎓 Learning Outcomes

Through this project, the following technical skills are demonstrated:

Machine Learning
Deep Learning
Regression
Feature Engineering
Data Preprocessing
Model Training
Model Evaluation
Python Programming
XGBoost
PyTorch
FastAPI Development
REST API Development
Frontend Development
AI-based Cost Prediction
Construction Data Analysis
🔍 Key Machine Learning Concepts

The project demonstrates:

Concept	Purpose
Regression	Predict continuous construction costs
Feature Engineering	Create useful features from raw data
Data Preprocessing	Prepare data for model training
XGBoost	Gradient boosting regression
PyTorch	Neural network-based prediction
Model Evaluation	Measure prediction performance
Train/Test Split	Evaluate model generalization
Chronological Validation	Simulate future prediction
🔄 End-to-End System
Construction Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Model Training
        ↓
XGBoost / PyTorch
        ↓
Model Evaluation
        ↓
Save Trained Model
        ↓
FastAPI Backend
        ↓
Web Application
        ↓
User Construction Parameters
        ↓
Prediction
        ↓
Estimated Construction Cost
🏠 Construction Cost Prediction

The system takes construction-related inputs from the user and processes them through the trained machine learning model.

Input Parameters
       ↓
Building Information
       ↓
Material Information
       ↓
Labour Information
       ↓
Feature Engineering
       ↓
Trained Model
       ↓
Predicted Construction Cost
📊 Example Prediction Process
User enters:

Construction Year
Plot Area
Built-up Area
Number of Floors
House Type
Construction Quality
Material Quantities
Labour Requirements
        ↓
System processes the inputs
        ↓
Feature engineering
        ↓
Machine learning prediction
        ↓
Estimated Construction Cost
🔮 Future Enhancements

Possible future improvements include:

Integration with live material prices
Location-specific construction cost prediction
Real-time market price updates
Cloud deployment
Database integration
Advanced visualization dashboards
Larger construction datasets
Improved deep learning architectures
Mobile application support
Automated PDF cost reports
Construction material price forecasting
Regional cost comparison
User authentication
Historical prediction tracking
📌 Advantages
Reduces manual calculation effort
Provides quick preliminary estimates
Uses historical construction information
Supports future cost prediction
Combines machine learning and deep learning
Provides a web-based interface
Can be integrated with other construction planning systems
Helps users make early budgeting decisions
⚠️ Limitations

The prediction depends on the quality and range of the historical dataset.

Actual construction costs may differ because of:

Material price fluctuations
Labour rate changes
Location
Transportation costs
Site conditions
Design modifications
Construction quality
Government taxes
Market conditions
Unexpected construction expenses
⚠️ Disclaimer

The predicted construction cost is an AI-based preliminary estimate.

Actual construction costs may vary depending on location, material prices, labour rates, design changes, site conditions, transportation costs, taxes, construction quality, and other real-world factors.

The prediction should therefore be used as a planning and budgeting reference and not as a final construction quotation.

🚀 Future Scope

The project can be extended into a complete AI-assisted construction planning platform by integrating:

Cost Prediction
      +
Material Estimation
      +
Labour Estimation
      +
Future Price Forecasting
      +
Residential Layout Planning
      +
3D Visualization
      +
Automated Cost Reports

This can help create an integrated intelligent system for residential construction planning and estimation.

📜 License

This project is developed for academic, educational, and research purposes.

👤 Author

Construction Price Prediction – AI Project

⭐ Conclusion

BuildCost AI demonstrates the integration of Machine Learning, Deep Learning, FastAPI, and web technologies to develop an intelligent residential construction cost prediction system.

By analyzing historical construction data and project parameters, the system provides data-driven preliminary construction cost estimates that can assist homeowners, builders, contractors, architects, and construction planners during the planning and budgeting stages.
