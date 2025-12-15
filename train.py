import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Minimal knobs for multi-user environments (e.g., shared research facilities)
# - MLFLOW_TRACKING_URI: set to a remote server if available; default is local ./mlruns
# - MLFLOW_EXPERIMENT_NAME: experiment name to group runs
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlflow-mlops-demo")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Carichiamo dataset
data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

with mlflow.start_run():
    mlflow.set_tag("use_case", "reproducible_training_tracking")
    mlflow.set_tag("dataset", "sklearn_diabetes")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    # Log parametri, metriche e modello
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("MSE:", mse)
    print("Modello salvato con MLflow!")