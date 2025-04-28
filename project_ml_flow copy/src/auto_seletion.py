import mlflow
from mlflow.tracking import MlflowClient

# 1. Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")  # <-- careful: ./mlruns, not ../mlruns

# 2. Define experiment
experiment_name = "Vestiaire_Model_Comparison"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise Exception(f"Experiment '{experiment_name}' not found!")

experiment_id = experiment.experiment_id
client = MlflowClient()

# 3. Search all runs
runs = client.search_runs(
    experiment_ids=[experiment_id],
    max_results=50
)

# 4. Find best run based on roc_auc
runs_with_roc = [r for r in runs if "roc_auc" in r.data.metrics]
runs_with_roc.sort(key=lambda r: r.data.metrics['roc_auc'], reverse=True)

# 5. Select best run
if runs_with_roc:
    best_run = runs_with_roc[0]
    best_run_id = best_run.info.run_id
    print(f" Best run ID: {best_run_id}")
    print(f"Best ROC AUC (Validation): {best_run.data.metrics['roc_auc']}")
else:
    raise Exception("No runs with roc_auc metric found!")

# 6. Load the best model automatically
best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
print("Best model loaded successfully!")
