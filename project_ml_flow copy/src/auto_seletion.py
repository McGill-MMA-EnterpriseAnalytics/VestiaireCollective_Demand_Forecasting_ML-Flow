import mlflow
from mlflow.tracking import MlflowClient
# Set MLflow experiment
import os

# other_mlruns_path = os.path.abspath("/Users/chloe/PycharmProjects/VestiaireCollective_Demand_Forecasting_ML-Flow/project_ml_flow copy/mlruns")
# tracking_uri = f"file://{other_mlruns_path}"


tracking_uri = f"file://{os.getcwd()}/mlruns"
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()
experiment_name = "Vestiaire_Model_Comparison"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found. Please make sure it exists.")

experiment_id = experiment.experiment_id
# Search all runs
runs = client.search_runs(
    experiment_ids=[experiment_id],
    max_results=50
)

# 2. Find the best run with roc_auc
runs_with_roc = [r for r in runs if "roc_auc" in r.data.metrics]

# Sort runs by roc_auc descending
runs_with_roc.sort(key=lambda r: r.data.metrics['roc_auc'], reverse=True)

# 3. Select best run
if runs_with_roc:
    best_run = runs_with_roc[0]
    best_run_id = best_run.info.run_id
    print(f"Best run ID: {best_run_id}")
    print(f"Best ROC AUC (Validation): {best_run.data.metrics['roc_auc']}")

    # Save best_run_id into a text file
    with open("../best_run_id.txt", "w") as f:
        f.write(best_run_id)

else:
    raise Exception("No runs with roc_auc metric found!")

