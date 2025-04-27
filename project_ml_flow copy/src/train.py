#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split, learning_curve, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, log_loss, roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import tempfile
import os

# Set MLflow experiment
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Vestiaire_Model_Comparison")
if mlflow.active_run():
    mlflow.end_run()

# Define core functions
def train_and_evaluate_with_mlflow(model, model_name, X_train, y_train, X_val, y_val, parent_run_id=None):
    with mlflow.start_run(run_name=model_name, nested=True, parent_run_id=parent_run_id):
        start_time = time.time()

        # Train and predict on validation set
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        # Compute metrics and log
        roc_auc = roc_auc_score(y_val, y_pred_prob)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred_prob)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "log_loss": logloss
        })

        # Log ROC & PR curves as artifacts
        temp_dir = tempfile.mkdtemp()
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
        plt.figure(); plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})"); plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f"{model_name} ROC (Validation)"); plt.legend()
        roc_path = os.path.join(temp_dir, f"{model_name}_validation_roc.png")
        plt.savefig(roc_path); mlflow.log_artifact(roc_path); plt.close()

        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_prob)
        plt.figure(); plt.plot(recall_vals, precision_vals, label="PR Curve"); plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f"{model_name} PR (Validation)"); plt.legend()
        prc_path = os.path.join(temp_dir, f"{model_name}_validation_prc.png")
        plt.savefig(prc_path); mlflow.log_artifact(prc_path); plt.close()

        return model


def plot_learning_curve_and_log(model, X_train, y_train, model_name):
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1,
        train_sizes=np.linspace(0.1, 0.5, 3)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label=f"{model_name} Train")
    plt.plot(train_sizes, test_mean, label=f"{model_name} Validation")
    plt.title(f"{model_name} Learning Curve"); plt.xlabel("Training Size"); plt.ylabel("ROC-AUC Score"); plt.legend(); plt.grid()

    temp_dir = tempfile.mkdtemp()
    plot_path = os.path.join(temp_dir, f"{model_name}_learning_curve.png")
    plt.savefig(plot_path); mlflow.log_artifact(plot_path); plt.close()

# Initialize models
xgb_model = xgb.XGBClassifier(eval_metric='auc', random_state=42, n_estimators=30, max_depth=3, tree_method='hist', n_jobs=-1)

lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=30, max_depth=3, device='cpu')

cat_model = cb.CatBoostClassifier(verbose=0, random_state=42, iterations=30, depth=3, task_type='CPU')

# Entry point
if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_parquet('../data/cleaned_data.parquet')
    # Feature engineering
    df['margin_rate'] = (df['seller_price'] - df['seller_earning']) / df['seller_price']
    df.drop(columns=['product_color', 'product_id'], inplace=True)
    df.drop(columns=['product_category_encoded', 'has_cross_border_fees_encoded'], inplace=True)
    df1 = df
    df1['price_to_earning_ratio'] = df1['price_usd'] / (df1['seller_earning'] + 1)
    df1['price_per_like'] = df1['price_usd'] / (df1['product_like_count'] + 1)
    df1['seller_activity_ratio'] = df1['seller_products_sold'] / (df1['seller_num_products_listed'] + 1)

    # Select features and target
    important_features = [
        'seller_price', 'seller_badge_encoded', 'should_be_gone', 'seller_pass_rate',
        'price_to_earning_ratio', 'seller_products_sold', 'price_per_like', 'brand_id',
        'product_type', 'product_material', 'product_like_count', 'seller_num_products_listed',
        'seller_community_rank', 'seller_activity_ratio', 'product_color_encoded',
        'seller_num_followers', 'margin_rate', 'available', 'seller_country', 'in_stock',
        'product_season_encoded', 'usually_ships_within_encoded', 'product_condition_encoded',
        'warehouse_name_encoded'
    ]
    X = df1[important_features]
    y = df1['sold']

    # Sample and split
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.3, random_state=42, stratify=y)
    X_train, X_temp, y_train, y_temp = train_test_split(X_sample, y_sample, test_size=0.4, random_state=42, stratify=y_sample)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Run experiments
    with mlflow.start_run(run_name="Model Comparison") as parent_run:
        parent_id = parent_run.info.run_id
        xgb_model = train_and_evaluate_with_mlflow(xgb_model, "XGBoost", X_train, y_train, X_val, y_val, parent_id)
        plot_learning_curve_and_log(xgb_model, X_train, y_train, "XGBoost")

        lgb_model = train_and_evaluate_with_mlflow(lgb_model, "LightGBM", X_train, y_train, X_val, y_val, parent_id)
        plot_learning_curve_and_log(lgb_model, X_train, y_train, "LightGBM")

        cat_model = train_and_evaluate_with_mlflow(cat_model, "CatBoost", X_train, y_train, X_val, y_val, parent_id)
        plot_learning_curve_and_log(cat_model, X_train, y_train, "CatBoost")

        # Final test evaluation
        with mlflow.start_run(run_name="Test_Evaluation", nested=True, parent_run_id=parent_id):
            for model, name in [(xgb_model, "XGBoost"), (lgb_model, "LightGBM"), (cat_model, "CatBoost")]:
                prob = model.predict_proba(X_test)[:, 1]
                pred = model.predict(X_test)
                test_metrics = {
                    f"{name}_test_roc_auc": roc_auc_score(y_test, prob),
                    f"{name}_test_accuracy": accuracy_score(y_test, pred),
                    f"{name}_test_precision": precision_score(y_test, pred),
                    f"{name}_test_recall": recall_score(y_test, pred),
                    f"{name}_test_f1": f1_score(y_test, pred),
                    f"{name}_test_log_loss": log_loss(y_test, prob)
                }
                mlflow.log_metrics(test_metrics)
                # log curves similarly as above

    print("Training and evaluation complete.")
