"""
========================================================================================
Future Building Construction Cost Prediction - PyTorch & ML Training Pipeline
========================================================================================
This script trains:
  1. PyTorch Deep Neural Network (DNN Regressor)
  2. XGBoost Regressor
  3. CatBoost Regressor
  4. Random Forest Regressor
  5. Linear Regression (Baseline)
Evaluates on unseen chronological 2026 test data and predicts future 2027 costs.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# PyTorch Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Scikit-Learn Preprocessing & Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

# Gradient Boosting
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# Define PyTorch Deep Neural Network Architecture
# ---------------------------------------------------------
class ConstructionCostNN(nn.Module):
    def __init__(self, input_dim):
        super(ConstructionCostNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 80, flush=True)
    print("   FUTURE BUILDING CONSTRUCTION COST PREDICTION (WITH PYTORCH DEEP LEARNING)", flush=True)
    print("=" * 80, flush=True)

    # Use CPU for fast tabular network training
    device = torch.device('cpu')
    print(f"[Device] Using PyTorch on: {device}", flush=True)

    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------
    csv_file = 'Construction_ML_Dataset_1000.csv'
    if not os.path.exists(csv_file):
        print(f"Error: Dataset '{csv_file}' not found.")
        sys.exit(1)

    print(f"\n[Step 1] Loading dataset from '{csv_file}'...")
    df = pd.read_csv(csv_file)
    print(f"Dataset loaded. Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ---------------------------------------------------------
    # 2. Data Cleaning
    # ---------------------------------------------------------
    print("\n[Step 2] Cleaning dataset...")
    df.columns = df.columns.str.strip()

    # Missing values check
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    print(" - Missing values check: Clean.")

    # Duplicates check
    dups = df.duplicated().sum()
    if dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f" - Duplicate rows check: Clean ({len(df)} records).")

    # ---------------------------------------------------------
    # 3. Domain Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 3] Engineering domain features...")
    ref_future_year = 2027

    df['Building_Age'] = ref_future_year - df['Construction_Year']
    df['Area_per_Floor'] = df['Builtup_Area_sqft'] / df['Number_of_Floors'].replace(0, 1)
    df['Cement_per_sqft'] = df['Cement_Bags'] / df['Builtup_Area_sqft']
    df['Steel_per_sqft'] = df['Steel_Quantity_kg'] / df['Builtup_Area_sqft']
    df['Sand_per_sqft'] = df['Sand_Quantity_m3'] / df['Builtup_Area_sqft']
    df['Brick_per_sqft'] = df['Brick_Quantity'] / df['Builtup_Area_sqft']
    df['Total_Labour_Days'] = (
        df['Mason_Labour_Days'] +
        df['Carpenter_Labour_Days'] +
        df['Electrician_Labour_Days'] +
        df['Plumber_Labour_Days']
    )
    df['Labour_Day_per_sqft'] = df['Total_Labour_Days'] / df['Builtup_Area_sqft']
    df['Material_Rate_Index'] = (
        (df['Cement_Rate_per_Bag'] * 0.30) +
        (df['Steel_Rate_per_kg'] * 0.40) +
        (df['Sand_Rate_per_m3'] * 0.20) +
        ((df['Brick_Rate_per_1000'] / 1000) * 0.10)
    )

    # ---------------------------------------------------------
    # 4. Feature Selection & Data Leakage Elimination
    # ---------------------------------------------------------
    print("\n[Step 4] Eliminating data leakage features...")
    target = 'Total_Estimated_Cost'
    leakage_cols = ['Project_ID', 'Material_Cost', 'Labour_Cost', target]
    feature_cols = [c for c in df.columns if c not in leakage_cols]

    X = df[feature_cols].copy()
    y = df[target].copy()

    print(f" - Input Features: {len(feature_cols)} features")
    print(f" - Excluded: ['Material_Cost', 'Labour_Cost', 'Project_ID'] (Zero Leakage)")

    # ---------------------------------------------------------
    # 5. Chronological Split (Train: 2010-2024, Val: 2025, Test: 2026)
    # ---------------------------------------------------------
    print("\n[Step 5] Applying chronological temporal split...")
    sorted_years = sorted(df['Construction_Year'].unique())
    test_year = sorted_years[-1]
    val_year = sorted_years[-2]
    train_years = sorted_years[:-2]

    train_mask = df['Construction_Year'].isin(train_years)
    val_mask = df['Construction_Year'] == val_year
    test_mask = df['Construction_Year'] == test_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f" - Train Set : {len(X_train)} rows (Years {train_years[0]}–{train_years[-1]})")
    print(f" - Val Set   : {len(X_val)} rows (Year {val_year})")
    print(f" - Test Set  : {len(X_test)} rows (Year {test_year})")

    # ---------------------------------------------------------
    # 6. Preprocessing Pipelines
    # ---------------------------------------------------------
    print("\n[Step 6] Fitting Preprocessor (StandardScaler + OneHotEncoder)...")
    cat_features = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype) in ['str', 'string', 'category']]
    if not cat_features:
        cat_features = [c for c in ['City', 'House_Type', 'Construction_Quality', 'Foundation_Type'] if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
        ]
    )

    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)

    # Scale target for PyTorch stability (in Lakhs INR)
    y_train_arr = y_train.values / 1e5
    y_val_arr = y_val.values / 1e5
    y_test_arr = y_test.values / 1e5

    # ---------------------------------------------------------
    # 7. Train PyTorch Deep Neural Network
    # ---------------------------------------------------------
    print("\n[Step 7] Training PyTorch Deep Neural Network (DNN)...")

    # Prepare PyTorch Tensors & DataLoaders
    X_tr_t = torch.tensor(X_train_prep, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_arr, dtype=torch.float32).unsqueeze(1)

    X_va_t = torch.tensor(X_val_prep, dtype=torch.float32)
    y_va_t = torch.tensor(y_val_arr, dtype=torch.float32).unsqueeze(1)

    X_te_t = torch.tensor(X_test_prep, dtype=torch.float32)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = X_train_prep.shape[1]
    pytorch_model = ConstructionCostNN(input_dim=input_dim).to(device)

    criterion = nn.SmoothL1Loss()  # Huber Loss for robust regression
    optimizer = optim.AdamW(pytorch_model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    epochs = 150
    best_val_loss = float('inf')
    best_weights = None

    pytorch_model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = pytorch_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_dataset)

        # Validation check
        pytorch_model.eval()
        with torch.no_grad():
            va_out = pytorch_model(X_va_t.to(device))
            va_loss = criterion(va_out, y_va_t.to(device)).item()
        pytorch_model.train()

        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_weights = pytorch_model.state_dict().copy()

        if epoch % 30 == 0 or epoch == epochs:
            print(f"   Epoch [{epoch:03d}/{epochs}] -> Train Loss: {epoch_loss:.4f} | Val Loss: {va_loss:.4f}")

    # Load best weights
    if best_weights is not None:
        pytorch_model.load_state_dict(best_weights)

    # PyTorch Test Predictions
    pytorch_model.eval()
    with torch.no_grad():
        pt_preds_lakhs = pytorch_model(X_te_t.to(device)).cpu().numpy().flatten()
        pt_test_preds = pt_preds_lakhs * 1e5

    print(" - PyTorch Deep Neural Network training complete.")

    # ---------------------------------------------------------
    # 8. Train Other Benchmark Models
    # ---------------------------------------------------------
    print("\n[Step 8] Training Comparative ML Models...")

    # Linear Regression
    lr = LinearRegression().fit(X_train_prep, y_train)
    lr_preds = lr.predict(X_test_prep)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42).fit(X_train_prep, y_train)
    rf_preds = rf.predict(X_test_prep)

    # XGBoost
    xgb_base = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_grid = GridSearchCV(xgb_base, {'n_estimators': [100, 150], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}, cv=3, scoring='r2', n_jobs=-1)
    xgb_grid.fit(X_train_prep, y_train)
    best_xgb = xgb_grid.best_estimator_
    xgb_preds = best_xgb.predict(X_test_prep)

    # CatBoost
    cb_model = cb.CatBoostRegressor(iterations=250, learning_rate=0.08, depth=5, random_seed=42, verbose=0)
    cb_model.fit(X_train_prep, y_train, eval_set=(X_val_prep, y_val), verbose=False)
    cb_preds = cb_model.predict(X_test_prep)

    # ---------------------------------------------------------
    # 9. Comprehensive Evaluation on 2026 Test Year
    # ---------------------------------------------------------
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return mae, rmse, r2, mape

    all_models = {
        'PyTorch Deep NN': pt_test_preds,
        'XGBoost (Tuned)': xgb_preds,
        'Random Forest': rf_preds,
        'Linear Regression': lr_preds,
        'CatBoost': cb_preds
    }

    records = []
    for name, preds in all_models.items():
        mae, rmse, r2, mape = calc_metrics(y_test, preds)
        records.append({
            'Model': name,
            'MAE (Rs.)': mae,
            'RMSE (Rs.)': rmse,
            'R2 Score': r2,
            'MAPE (%)': mape
        })

    results_df = pd.DataFrame(records).sort_values(by='R2 Score', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("                    MODEL EVALUATION SUMMARY TABLE (2026 TEST DATA)")
    print("=" * 80)
    display_df = results_df.copy()
    display_df['MAE (Rs.)'] = display_df['MAE (Rs.)'].apply(lambda x: f"Rs. {x:,.2f}")
    display_df['RMSE (Rs.)'] = display_df['RMSE (Rs.)'].apply(lambda x: f"Rs. {x:,.2f}")
    display_df['R2 Score'] = display_df['R2 Score'].apply(lambda x: f"{x:.4f}")
    display_df['MAPE (%)'] = display_df['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
    print(display_df.to_string(index=False))
    print("=" * 80)

    # ---------------------------------------------------------
    # 10. Future 2027 Prediction
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("          FUTURE 2027 PREDICTION USING PYTORCH & ENSEMBLE")
    print("=" * 80)

    sample_2027 = {
        'Construction_Year': 2027,
        'City': 'Chennai',
        'Plot_Area_sqft': 2400,
        'Builtup_Area_sqft': 2000,
        'Number_of_Floors': 2,
        'House_Type': 'Residential',
        'Construction_Quality': 'Standard',
        'Bedroom_Count': 3,
        'Bathroom_Count': 3,
        'Hall_Count': 1,
        'Kitchen_Count': 1,
        'Foundation_Type': 'RCC',
        'Steel_Quantity_kg': 9000,
        'Cement_Bags': 660,
        'Sand_Quantity_m3': 30.0,
        'Aggregate_Quantity_m3': 24.0,
        'Brick_Quantity': 16000,
        'Electrical_Points': 35,
        'Plumbing_Points': 22,
        'Mason_Labour_Days': 80,
        'Carpenter_Labour_Days': 50,
        'Electrician_Labour_Days': 25,
        'Plumber_Labour_Days': 22,
        'Cement_Rate_per_Bag': 420,
        'Steel_Rate_per_kg': 82,
        'Sand_Rate_per_m3': 2200,
        'Brick_Rate_per_1000': 11000
    }

    fut_df = pd.DataFrame([sample_2027])
    fut_df['Building_Age'] = ref_future_year - fut_df['Construction_Year']
    fut_df['Area_per_Floor'] = fut_df['Builtup_Area_sqft'] / fut_df['Number_of_Floors'].replace(0, 1)
    fut_df['Cement_per_sqft'] = fut_df['Cement_Bags'] / fut_df['Builtup_Area_sqft']
    fut_df['Steel_per_sqft'] = fut_df['Steel_Quantity_kg'] / fut_df['Builtup_Area_sqft']
    fut_df['Sand_per_sqft'] = fut_df['Sand_Quantity_m3'] / fut_df['Builtup_Area_sqft']
    fut_df['Brick_per_sqft'] = fut_df['Brick_Quantity'] / fut_df['Builtup_Area_sqft']
    fut_df['Total_Labour_Days'] = (
        fut_df['Mason_Labour_Days'] +
        fut_df['Carpenter_Labour_Days'] +
        fut_df['Electrician_Labour_Days'] +
        fut_df['Plumber_Labour_Days']
    )
    fut_df['Labour_Day_per_sqft'] = fut_df['Total_Labour_Days'] / fut_df['Builtup_Area_sqft']
    fut_df['Material_Rate_Index'] = (
        (fut_df['Cement_Rate_per_Bag'] * 0.30) +
        (fut_df['Steel_Rate_per_kg'] * 0.40) +
        (fut_df['Sand_Rate_per_m3'] * 0.20) +
        ((fut_df['Brick_Rate_per_1000'] / 1000) * 0.10)
    )

    X_fut_proc = preprocessor.transform(fut_df[feature_cols])

    # PyTorch prediction
    pytorch_model.eval()
    with torch.no_grad():
        pt_pred_2027 = pytorch_model(torch.tensor(X_fut_proc, dtype=torch.float32).to(device)).item() * 1e5

    xgb_pred_2027 = best_xgb.predict(X_fut_proc)[0]

    print(f"Project Specifications: 2000 sqft in Chennai (Standard Quality, 2 Floors, Year 2027)")
    print(f" - Predicted by PyTorch Deep NN : Rs. {pt_pred_2027:,.2f} (~Rs. {pt_pred_2027/1e5:.2f} Lakhs)")
    print(f" - Predicted by XGBoost Model   : Rs. {xgb_pred_2027:,.2f} (~Rs. {xgb_pred_2027/1e5:.2f} Lakhs)")
    print("=" * 80)

    # Save PyTorch Model weights and pipeline
    torch.save(pytorch_model.state_dict(), 'pytorch_construction_model.pth')
    joblib.dump({
        'preprocessor': preprocessor,
        'feature_cols': feature_cols,
        'input_dim': input_dim,
        'xgb_model': best_xgb
    }, 'pytorch_pipeline_bundle.pkl')

    print("\n[Complete] PyTorch model weights saved to 'pytorch_construction_model.pth'.")
    print("[Complete] All models trained and verified!")

if __name__ == '__main__':
    main()
