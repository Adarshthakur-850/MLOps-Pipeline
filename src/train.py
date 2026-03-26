import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes
import sys
import os

# Set MLflow tracking URI
# Default to local ./mlruns if MLFLOW_TRACKING_URI is not set
# This ensures it works out-of-the-box locally without docker
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
print(f"Logging to: {tracking_uri}")

def train():
    # Load Diabetes dataset (Regression task)
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameters
    n_estimators = 100
    max_depth = 10

    # Start MLflow Run
    mlflow.set_experiment("Diabetes_Prediction")
    
    with mlflow.start_run():
        # Train Model
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        # Predict
        predictions = rf.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5  # Manual RMSE for compatibility
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

        # Log Parameters & Metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log Model
        mlflow.sklearn.log_model(rf, "model", registered_model_name="DiabetesRFModel")
        print("Model logged to MLflow.")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed: {e}")
