import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATASET_PATH = "test.npy" 
MODEL_SAVE_PATH = "test.json"
VALIDATION_SPLIT = 0.2

def train_xgboost():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}.")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    real_data = np.load(DATASET_PATH)
    
    X = real_data[:, :521]
    y = real_data[:, -1]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print("\n--- Starting Aggressive XGBoost Training (Extended Run) ---")

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=5000,
        learning_rate=0.01,      
        max_depth=3,             
        min_child_weight=7,      
        subsample=0.7,           
        colsample_bytree=0.4,    
        gamma=2.0,               
        reg_alpha=1.0,           
        reg_lambda=3.0,          
        early_stopping_rounds=150, 
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100  
    )

    val_predictions = model.predict(X_val)
    best_val_mse = mean_squared_error(y_val, val_predictions)
    
    print(f"Training complete!")
    print(f"Best validation MSE: {best_val_mse:.2f}")
    
    model.save_model(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_xgboost()