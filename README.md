# MLflow MLOps Demo

Minimal example of **experiment tracking with MLflow**.

What it does:
- loads a regression dataset (scikit-learn diabetes)
- trains a `LinearRegression`
- logs parameters + metrics to MLflow
- logs the trained model as an artifact

## Quickstart

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run a training run (logs to `./mlruns` by default):

```bash
python train.py
```

Open the MLflow UI:

```bash
mlflow ui
```

Then visit `http://127.0.0.1:5000`.

## Config knobs (optional)

The script supports a few environment variables so it behaves better in shared environments:

```bash
# Example: group runs under a custom experiment
export MLFLOW_EXPERIMENT_NAME="facility-demo"

# Example: point to a remote MLflow tracking server
# export MLFLOW_TRACKING_URI="http://mlflow.your-lab.local:5000"

# Example: control split + reproducibility
export TEST_SIZE="0.2"
export RANDOM_STATE="42"

python train.py
```

## Why this matters in a research facility

In imaging-heavy labs, you often need a simple audit trail for processing and analysis:
- **Reproducibility**: log parameters and code versions indirectly via structured runs
- **Traceability**: compare metrics across script versions or hardware setups
- **Operational sanity**: make runs inspectable by others without passing around spreadsheets

This repo is intentionally small: it demonstrates the *pattern* (tracking + artifacts) without turning into a full platform.
