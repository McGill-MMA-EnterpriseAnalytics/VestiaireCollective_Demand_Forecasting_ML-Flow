"""
Price Elasticity Analysis - Part 2
Advanced Modeling with Boosting Methods and Class Imbalance Handling

This module continues the analysis started in Price_Elasticity_1stPart.py,
focusing on predictive modeling using boosting algorithms (XGBoost, LightGBM, 
and CatBoost) and handling class imbalance with SMOTETomek.

Key components:
1. Fast model comparison with simplified parameters
2. Model training with SMOTETomek for handling class imbalance
3. Model evaluation without seller_price to check for potential data leakage
4. Learning curve analysis to assess model performance

Dependencies:
- This module builds on the DataFrame prepared in Price_Elasticity_1stPart.py
- Requires the derived features created in the first part
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
from imblearn.combine import SMOTETomek


def get_important_features():
    """
    Return the list of important features for modeling.
    
    Returns:
        list: List of feature names identified as important
    """
    return [
        'seller_price', 'seller_badge_encoded', 'should_be_gone', 'seller_pass_rate',
        'price_to_earning_ratio', 'seller_products_sold', 'price_per_like', 'brand_id',
        'product_type', 'product_material', 'product_like_count', 'seller_num_products_listed',
        'seller_community_rank', 'seller_activity_ratio', 'product_color_encoded',
        'seller_num_followers', 'available', 'seller_country', 'in_stock',
        'product_season_encoded', 'usually_ships_within_encoded', 'product_condition_encoded',
        'warehouse_name_encoded'
    ]


def get_important_features_no_price():
    """
    Return the list of important features excluding seller_price.
    
    Returns:
        list: List of feature names excluding seller_price
    """
    features = get_important_features()
    features.remove('seller_price')
    return features


def prepare_data_sample(df, feature_list, target_col='sold', sample_size=0.3, random_state=42):
    """
    Prepare a sampled dataset for faster model training and evaluation.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_list (list): List of features to use
        target_col (str): Target column name
        sample_size (float): Proportion of data to sample
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_sample, y_sample, X_train, X_test, y_train, y_test)
    """
    # Select features and target
    X = df[feature_list]
    y = df[target_col]
    
    # Use a sample of the dataset for faster learning curves
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=sample_size, random_state=random_state, stratify=y
    )
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.3, random_state=random_state, stratify=y_sample
    )
    
    return X_sample, y_sample, X_train, X_test, y_train, y_test


def prepare_full_data(df, feature_list, target_col='sold', test_size=0.3, random_state=42):
    """
    Prepare the full dataset for model training and evaluation.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_list (list): List of features to use
        target_col (str): Target column name
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Select features and target
    X = df[feature_list]
    y = df[target_col]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def apply_smote_tomek(X_train, y_train, random_state=42):
    """
    Apply SMOTETomek to handle class imbalance.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train_resampled, y_train_resampled)
    """
    smote_tomek = SMOTETomek(random_state=random_state)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    # Print class distribution before and after resampling
    print("Original class distribution:")
    print(pd.Series(y_train).value_counts())
    print("Resampled class distribution:")
    print(pd.Series(y_train_resampled).value_counts())
    
    return X_train_resampled, y_train_resampled


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train a model and evaluate its performance using ROC-AUC.
    
    Args:
        model: The model to train
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        X_test (pandas.DataFrame): Testing features
        y_test (pandas.Series): Testing target
        
    Returns:
        model: The trained model
    """
    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    elapsed_time = (time.time() - start_time) / 60
    print(f"{model.__class__.__name__} ROC-AUC: {roc_auc:.4f} | Time: {elapsed_time:.2f} min")
    return model


def initialize_models(fast=False, random_state=42):
    """
    Initialize boosting models with appropriate parameters.
    
    Args:
        fast (bool): Whether to use faster, simplified models
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (xgb_model, lgb_model, cat_model)
    """
    if fast:
        # Simplified models for faster execution
        xgb_model = xgb.XGBClassifier(
            eval_metric='auc', random_state=random_state,
            n_estimators=30, max_depth=3, tree_method='hist', n_jobs=-1
        )
        
        lgb_model = lgb.LGBMClassifier(
            random_state=random_state, n_estimators=30, max_depth=3, device='cpu'
        )
        
        cat_model = cb.CatBoostClassifier(
            verbose=0, random_state=random_state, iterations=30, depth=3, task_type='CPU'
        )
    else:
        # Full models
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='auc', random_state=random_state,
            n_estimators=100, max_depth=5, n_jobs=-1
        )
        
        lgb_model = lgb.LGBMClassifier(
            random_state=random_state, n_estimators=100, max_depth=5, n_jobs=-1
        )
        
        cat_model = cb.CatBoostClassifier(
            verbose=0, random_state=random_state, iterations=100, depth=5
        )
    
    return xgb_model, lgb_model, cat_model


def plot_learning_curve(model, X, y, model_name, fast=False):
    """
    Plot the learning curve for a model.
    
    Args:
        model: Trained model
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        model_name (str): Name of the model for the plot
        fast (bool): Whether to use faster, simplified settings
        
    Returns:
        float: Time taken to plot the learning curve in minutes
    """
    start_time = time.time()

    if fast:
        # Faster settings
        cv = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
        train_sizes = np.linspace(0.1, 0.5, 3)  # Fewer training sizes
    else:
        # Normal settings
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
        train_sizes = np.linspace(0.1, 1.0, 5)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1,
        train_sizes=train_sizes
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, label=f"{model_name} Train")
    plt.plot(train_sizes, test_mean, label=f"{model_name} Test")

    elapsed_time = (time.time() - start_time) / 60
    print(f"{model_name} Learning Curve completed in {elapsed_time:.2f} min")
    
    return elapsed_time


def setup_plot(title):
    """
    Set up a new plot with the given title.
    
    Args:
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("ROC-AUC Score")
    plt.grid()


def finalize_plot():
    """
    Add legend and display the plot.
    """
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_feature_correlations(X_train, y_train):
    """
    Analyze correlations between features and the target variable.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        pandas.Series: Correlation values
    """
    correlations = X_train.corrwith(y_train)
    return correlations


def run_fast_comparison(df):
    """
    Run a fast comparison of the three boosting algorithms.
    
    Args:
        df (pandas.DataFrame): Input dataframe
    """
    print("\n--- Running Fast Model Comparison ---\n")
    
    # Get important features
    important_features = get_important_features()
    
    # Prepare data sample for faster learning curves
    X_sample, y_sample, X_train, X_test, y_train, y_test = prepare_data_sample(
        df, important_features
    )
    
    # Initialize models with simplified parameters
    xgb_model, lgb_model, cat_model = initialize_models(fast=True)
    
    # Train and evaluate models
    xgb_model = train_and_evaluate(xgb_model, X_train, y_train, X_test, y_test)
    lgb_model = train_and_evaluate(lgb_model, X_train, y_train, X_test, y_test)
    cat_model = train_and_evaluate(cat_model, X_train, y_train, X_test, y_test)
    
    # Plot learning curves
    setup_plot("Learning Curves Comparison")
    plot_learning_curve(xgb_model, X_sample, y_sample, "XGBoost", fast=True)
    plot_learning_curve(lgb_model, X_sample, y_sample, "LightGBM", fast=True)
    plot_learning_curve(cat_model, X_sample, y_sample, "CatBoost", fast=True)
    finalize_plot()


def run_with_smote_tomek(df):
    """
    Run models with SMOTETomek to address class imbalance.
    
    Args:
        df (pandas.DataFrame): Input dataframe
    """
    print("\n--- Running Models with SMOTETomek ---\n")
    
    # Get important features
    important_features = get_important_features()
    
    # Prepare full data
    X_train, X_test, y_train, y_test = prepare_full_data(df, important_features)
    
    # Apply SMOTETomek
    X_train_resampled, y_train_resampled = apply_smote_tomek(X_train, y_train)
    
    # Initialize models
    xgb_model, lgb_model, cat_model = initialize_models(fast=False)
    
    # Train and evaluate models with resampled data
    xgb_model = train_and_evaluate(xgb_model, X_train_resampled, y_train_resampled, X_test, y_test)
    lgb_model = train_and_evaluate(lgb_model, X_train_resampled, y_train_resampled, X_test, y_test)
    cat_model = train_and_evaluate(cat_model, X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Plot learning curves
    setup_plot("Learning Curves Comparison (with SMOTETomek)")
    plot_learning_curve(xgb_model, X_train_resampled, y_train_resampled, "XGBoost")
    plot_learning_curve(lgb_model, X_train_resampled, y_train_resampled, "LightGBM")
    plot_learning_curve(cat_model, X_train_resampled, y_train_resampled, "CatBoost")
    finalize_plot()
    
    return X_train, y_train, X_test, y_test, xgb_model, lgb_model, cat_model


def run_without_seller_price(df):
    """
    Run models without seller_price to reduce potential overfitting.
    
    Args:
        df (pandas.DataFrame): Input dataframe
    """
    print("\n--- Running Models Without Seller Price ---\n")
    
    # Get important features excluding seller_price
    important_features = get_important_features_no_price()
    
    # Prepare full data
    X_train, X_test, y_train, y_test = prepare_full_data(df, important_features)
    
    # Apply SMOTETomek
    X_train_resampled, y_train_resampled = apply_smote_tomek(X_train, y_train)
    
    # Initialize models
    xgb_model, lgb_model, cat_model = initialize_models(fast=False)
    
    # Train and evaluate models with resampled data
    xgb_model = train_and_evaluate(xgb_model, X_train_resampled, y_train_resampled, X_test, y_test)
    lgb_model = train_and_evaluate(lgb_model, X_train_resampled, y_train_resampled, X_test, y_test)
    cat_model = train_and_evaluate(cat_model, X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Plot learning curves
    setup_plot("Learning Curves Comparison (with SMOTETomek, without seller_price)")
    plot_learning_curve(xgb_model, X_train_resampled, y_train_resampled, "XGBoost")
    plot_learning_curve(lgb_model, X_train_resampled, y_train_resampled, "LightGBM")
    plot_learning_curve(cat_model, X_train_resampled, y_train_resampled, "CatBoost")
    finalize_plot()
    
    # Check for data leakage via feature correlations
    correlations = analyze_feature_correlations(X_train, y_train)
    print("\nFeature Correlations with Target:")
    print(correlations)


def main(df):
    """
    Main function to run all parts of the analysis.
    
    Args:
        df (pandas.DataFrame): The dataframe from the first part of the analysis
    """
    # Run fast comparison
    run_fast_comparison(df)
    
    # Run with SMOTETomek
    X_train, y_train, X_test, y_test, xgb_model, lgb_model, cat_model = run_with_smote_tomek(df)
    
    # Run without seller_price
    run_without_seller_price(df)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    # If this script is run directly, load the data from the first part
    # Assuming df1 was created and saved in the first part
    try:
        # Attempt to import from the first part
        from Price_Elasticity_1stPart import create_derived_features, load_data
        
        # Load and prepare data
        df = load_data('/Users/berlybrigith/Downloads/WINTER SEM/WINTER 2/IS Production 2/PROJECT/Folder structure for unit testing/data')
        df = create_derived_features(df)
        
        # Run the main analysis
        main(df)
    except ImportError:
        print("Error: Please run Price_Elasticity_1stPart.py first or ensure it's in the same directory.")