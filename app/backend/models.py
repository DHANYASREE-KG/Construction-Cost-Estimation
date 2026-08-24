import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# PyTorch Deep Neural Network definition
class ConstructionCostNN(nn.Module):
    def __init__(self, input_dim: int):
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

# Pydantic Schemas for API
class PredictionRequest(BaseModel):
    Construction_Year: int = Field(2027, description="Target future construction year")
    City: str = Field("Chennai", description="City location")
    Plot_Area_sqft: float = Field(2400.0, description="Total plot area in sqft")
    Builtup_Area_sqft: float = Field(2000.0, description="Total builtup area in sqft")
    Number_of_Floors: int = Field(2, description="Number of floors")
    House_Type: str = Field("Residential", description="Type of house")
    Construction_Quality: str = Field("Standard", description="Quality: Economy, Standard, Premium, Luxury")
    Bedroom_Count: int = Field(3, description="Number of bedrooms")
    Bathroom_Count: int = Field(3, description="Number of bathrooms")
    Hall_Count: int = Field(1, description="Number of halls")
    Kitchen_Count: int = Field(1, description="Number of kitchens")
    Foundation_Type: str = Field("RCC", description="Foundation structure")
    
    # Material Quantities
    Steel_Quantity_kg: float = Field(9000.0, description="Steel required in kg")
    Cement_Bags: float = Field(660.0, description="Cement required in bags")
    Sand_Quantity_m3: float = Field(30.0, description="Sand required in m3")
    Aggregate_Quantity_m3: float = Field(24.0, description="Aggregate required in m3")
    Brick_Quantity: float = Field(16000.0, description="Bricks required in units")
    
    # Fittings & Labour Days
    Electrical_Points: int = Field(35, description="Electrical points count")
    Plumbing_Points: int = Field(22, description="Plumbing points count")
    Mason_Labour_Days: float = Field(80.0, description="Mason labor days")
    Carpenter_Labour_Days: float = Field(50.0, description="Carpenter labor days")
    Electrician_Labour_Days: float = Field(25.0, description="Electrician labor days")
    Plumber_Labour_Days: float = Field(22.0, description="Plumber labor days")
    
    # Material Rates
    Cement_Rate_per_Bag: float = Field(420.0, description="Rate per bag in INR")
    Steel_Rate_per_kg: float = Field(82.0, description="Rate per kg in INR")
    Sand_Rate_per_m3: float = Field(2200.0, description="Rate per m3 in INR")
    Brick_Rate_per_1000: float = Field(11000.0, description="Rate per 1000 bricks in INR")

class PredictionResponse(BaseModel):
    status: str
    target_year: int
    predicted_cost_xgb: float
    predicted_cost_pytorch: float
    predicted_cost_rf: float
    predicted_cost_lr: float
    best_estimate_inr: str
    best_estimate_lakhs: str
    cost_per_sqft: str
    breakdown: Dict[str, float]
    selected_model: str
