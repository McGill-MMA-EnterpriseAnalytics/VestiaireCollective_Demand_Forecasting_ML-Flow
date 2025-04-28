import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:../mlruns")
experiment_name = "Vestiaire_Model_Comparison"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

client = MlflowClient()

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
    best_run_id = best_run.info.run_id  # <<< missing in your code!!!
    print(f"✅ Best run ID: {best_run_id}")
    print(f"✅ Best ROC AUC (Validation): {best_run.data.metrics['roc_auc']}")
else:
    raise Exception("❌ No runs with roc_auc metric found!")

# 4. Load the best model
best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
