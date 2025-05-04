# Demand-Forecasting-Vestiaire
Welcome! If you're intrigued by the evolving world of pre-owned luxury fashion, you're in the right place.

This project delves into Vestiaire Collective, a leading global platform for buying and selling second-hand designer fashion. By analyzing market trends, consumer behavior, seller dynamics, and price elasticity, we aim to uncover actionable insights that drive data-informed strategies for business growth and success.

Join us on this journey to explore the future of sustainable luxury fashion! 🌍👗


Link to company : https://us.vestiairecollective.com  
Link to dataset : https://www.kaggle.com/datasets/justinpakzad/vestiaire-fashion-dataset
---
## 🔍 Unit Testing

The tests/ directory contains unit tests for two key analytical components of the project:

- *Price Elasticity Unit Tests: These tests validate the functionality derived from the *causal analysis notebook focused on price elasticity. The original notebook has been modularized into three Python scripts to separate concerns and simplify testing:
  - Price_Elasticity_1stPart.py
  - Price_Elasticity_2ndPart.py
  - Price_Elasticity_3rdPart.py

- *Seller Analysis Tests: These cover logic and feature engineering from the *seller behavior analysis notebook.

Each test folder includes the relevant scripts and supporting documentation on how to implement and extend the unit testing process.

## Machine Learning Experiment Tracking with MLflow & Optuna

- sold_train.ipynb
- train.py

This section of our project implements comprehensive machine learning experiment tracking and hyperparameter optimization using MLflow and Optuna. These powerful tools help us manage our model development lifecycle, compare model performance, and find optimal hyperparameters efficiently.

### What are MLflow and Optuna?

- **MLflow** is an open-source platform that helps manage the entire machine learning lifecycle, including experimentation, reproducibility, deployment, and a central model registry.
- **Optuna** is a hyperparameter optimization framework specifically designed for machine learning, using Bayesian optimization to efficiently search the hyperparameter space.

### Implementation Overview

Our implementation uses MLflow and Optuna to:

1. **Track experiments** across multiple model types (XGBoost, LightGBM, CatBoost)
2. **Optimize hyperparameters** using Bayesian optimization
3. **Visualize model performance** through metrics and plots
4. **Compare models** across various evaluation metrics
5. **Document the model development process** for reproducibility

### MLflow Experiment Structure

Our MLflow implementation uses a hierarchical structure of runs:

```
Model_Comparison (Parent Run)
├── Hyperparameter_Tuning
│   ├── trial_0
│   ├── trial_1
│   └── ...
├── XGBoost
├── LightGBM
├── CatBoost
└── Test_Evaluation
```

### Hyperparameter Optimization with Optuna

We use Optuna to find optimal hyperparameters for our XGBoost model through Bayesian optimization:

```python
def objective(trial):
    # Sample hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "max_depth":    trial.suggest_int("max_depth",  3,  10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "subsample":    trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    
    # Train and evaluate model with these parameters
    # ...
    
    return auc  # Return metric to optimize
```

Each trial tests a different set of hyperparameters, and Optuna uses the results to guide the search toward promising areas of the parameter space.

![image](https://github.com/user-attachments/assets/d50acc9d-5c44-4bbb-81c3-e5a998105b7a)


### MLflow Tracking

For each model, we track:

#### Parameters
- Model-specific hyperparameters
- Training configuration
![image](https://github.com/user-attachments/assets/6312b287-71e7-4016-baa5-b03114d50214)


#### Metrics
- ROC-AUC score
- Accuracy
- Precision
- Recall
- F1 score
- Log loss

![image](https://github.com/user-attachments/assets/75dee077-6025-4f2b-919f-4cfd1eeab7b2)


#### Artifacts
1. **Performance Visualizations**
   - ROC curves
   - Precision-Recall curves
   - Confusion matrices
   - Learning curves

2. **Model Interpretability**
   - SHAP feature importance plots

3. **Sample Predictions**
   - CSV files with sample predictions for validation

![image](https://github.com/user-attachments/assets/42b636ed-39a7-4cba-b782-d1215020522e)
![image](https://github.com/user-attachments/assets/4e0b2453-cdd9-40e1-9779-1f4096c602a6)
![image](https://github.com/user-attachments/assets/e3d80cee-9b52-452c-8803-15bdfca17893)


### Key Functions

#### `train_and_evaluate_with_mlflow()`

This function:
- Trains a model on the training data
- Evaluates it on the validation set
- Logs parameters, metrics, and artifacts to MLflow
- Creates visualizations for model performance

#### `plot_learning_curve_and_log()`

This function:
- Generates learning curves to assess model performance with different training set sizes
- Helps identify if the model would benefit from more training data
- Logs the visualization to MLflow

### Comparing Models

The final evaluation on the held-out test set allows for direct comparison between models:

```python
# Final evaluation on the held-out test set
with mlflow.start_run(run_name="Test_Evaluation", nested=True, parent_run_id=parent_run_id):
    for model, model_name in [(xgb_model, "XGBoost"), (lgb_model, "LightGBM"), (cat_model, "CatBoost")]:
        y_test_pred_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            f"{model_name}_test_roc_auc": roc_auc_score(y_test, y_test_pred_prob),
            # Other metrics...
        }
        mlflow.log_metrics(metrics)
```

### How to View Experiment Results

1. Start the MLflow UI:
   ```bash
   mlflow ui --port 5000
   ```

2. Open your browser and navigate to http://localhost:5000

3. Select the "Vestiaire_Model_Comparison" experiment to view all runs

4. Compare models, analyze performance metrics, and view artifacts

### Benefits of Our Implementation

- **Reproducibility**: All experiments are tracked with their parameters and results
- **Efficiency**: Optuna's Bayesian optimization finds good hyperparameters with fewer trials
- **Visualization**: Comprehensive visualizations help understand model performance
- **Comparison**: Easy comparison between different model types
- **Documentation**: Automatic logging provides a history of the model development process


---

## 🤖 AutoML Implementation (Azure)

The `auto-ml/` directory contains a notebook implementing automated machine learning (AutoML) using **Azure Machine Learning**.

- **Notebook**: `AutoML_SoldItems_FINAL.ipynb`
- **Objective**: Automatically identify the best model and hyperparameters for predicting item demand (i.e., whether an item is sold).
- **Platform**: [Azure ML](https://azure.microsoft.com/en-us/products/machine-learning/)

### Features:
- Automated feature preprocessing and scaling
- Evaluation across multiple algorithms (e.g., LightGBM, XGBoost, Logistic Regression)
- Selection of the best run based on validation metrics (e.g., AUC, Accuracy)
- Integrated MLflow tracking for reproducibility
- Export of the best model pipeline for downstream use

📂 Path:  
`project_ml_flow copy/auto-ml/AutoML_SoldItems_FINAL.ipynb`

> This serves as a baseline reference to compare custom model performance against Azure's AutoML capabilities.
