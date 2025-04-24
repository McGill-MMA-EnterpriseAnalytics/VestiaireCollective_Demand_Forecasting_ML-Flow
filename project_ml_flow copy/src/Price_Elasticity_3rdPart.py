"""
Price Elasticity Analysis - Part 3
Causal Inference Analysis

This module continues the analysis from Price_Elasticity_1stPart.py and Price_Elasticity_2ndPart.py,
focusing on causal inference to analyze price elasticity effects using methods like:
- DoubleML for causal estimation
- Uplift modeling with TwoModels approach
- Treatment effect visualization

Key components:
1. Causal inference with DoubleML for price elasticity
2. Alternative treatment analysis with seller_price
3. Uplift modeling with binary treatment
4. Brand-based causal analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklift.models import TwoModels
from sklift.metrics import uplift_curve


def get_control_variables():
    """
    Get the list of control variables (X) for causal analysis.
    
    Returns:
        list: List of control variables
    """
    return [
        'product_type', 'product_category', 'product_season', 'product_condition',
        'seller_badge', 'has_cross_border_fees', 'seller_pass_rate',
        'price_per_like', 'price_to_earning_ratio', 'seller_activity_ratio'
    ]


def ensure_control_variables(df, control_vars):
    """
    Ensure all control variables exist in the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        control_vars (list): List of control variables
        
    Returns:
        list: Filtered list of control variables that exist in the dataframe
    """
    return [col for col in control_vars if col in df.columns]


def prepare_doubleml_data(df, outcome_var, treatment_var, control_vars):
    """
    Prepare data for DoubleML analysis.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        outcome_var (str): Outcome variable name (Y)
        treatment_var (str): Treatment variable name (T)
        control_vars (list): List of control variables (X)
        
    Returns:
        doubleml.DoubleMLData: Prepared data object for DoubleML
    """
    # Ensure no missing values in Y and T
    df_clean = df.copy()
    df_clean.dropna(subset=[outcome_var, treatment_var], inplace=True)
    
    # Create DoubleML data object
    data = DoubleMLData(df_clean, y_col=outcome_var, d_cols=treatment_var, x_cols=control_vars)
    
    return data


def train_doubleml_model(data, n_estimators=100, max_depth=6):
    """
    Train a DoubleML model for partial linear regression.
    
    Args:
        data (doubleml.DoubleMLData): Prepared data object
        n_estimators (int): Number of estimators for XGBoost
        max_depth (int): Maximum depth for XGBoost trees
        
    Returns:
        tuple: (model, coefficient, standard_error)
    """
    # Initialize regressors for nuisance functions
    ml_l = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, max_depth=max_depth)
    ml_m = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, max_depth=max_depth)
    
    # Create and fit DoubleML model
    model = DoubleMLPLR(data, ml_l, ml_m)
    model.fit()
    
    return model, model.coef, model.se


def interpret_elasticity(coefficient):
    """
    Interpret the price elasticity coefficient.
    
    Args:
        coefficient (float): Estimated price elasticity coefficient
        
    Returns:
        str: Interpretation of price elasticity
    """
    if coefficient < 0:
        return "Demand decreases as price increases (Elastic)."
    else:
        return "Demand increases with price (Inelastic)."


def plot_treatment_effects(df, treatment_var, treatment_effects, title):
    """
    Plot treatment effects against the treatment variable.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        treatment_var (str): Treatment variable name
        treatment_effects (numpy.ndarray): Estimated treatment effects
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of treatment effects vs treatment
    plt.scatter(df[treatment_var], treatment_effects, color='b', alpha=0.5)
    
    # Plot formatting
    plt.title(title, fontsize=16)
    plt.xlabel(f'{treatment_var}', fontsize=14)
    plt.ylabel('Treatment Effect (Uplift)', fontsize=14)
    plt.grid(True)
    
    return plt.gcf()


def create_price_categories(df, price_col='price_usd', n_categories=3):
    """
    Categorize prices into bins for uplift modeling.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        price_col (str): Price column name
        n_categories (int): Number of categories to create
        
    Returns:
        pandas.DataFrame: Dataframe with added price categories
    """
    df_result = df.copy()
    
    # Create price categories
    df_result['price_category'] = pd.qcut(df_result[price_col], q=n_categories, labels=range(n_categories))
    
    # Create binary high/low price indicator
    df_result['price_binary'] = (df_result['price_category'] > (n_categories - 2)).astype(int)
    
    return df_result


def prepare_uplift_data(df, X_cols, Y_col='sold', T_col='price_binary'):
    """
    Prepare data for uplift modeling.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        X_cols (list): Feature columns
        Y_col (str): Outcome variable
        T_col (str): Treatment variable
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, T_train, T_test)
    """
    # Ensure binary outcome
    df_prep = df.copy()
    df_prep[Y_col] = (df_prep[Y_col] > 0).astype(int)
    
    # Convert categorical features
    for col in X_cols:
        if df_prep[col].dtype == 'object':
            df_prep[col] = LabelEncoder().fit_transform(df_prep[col])
    
    # Split data
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        df_prep[X_cols], df_prep[Y_col], df_prep[T_col], 
        test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, T_train, T_test


def train_uplift_model(X_train, y_train, T_train, n_estimators=100):
    """
    Train a two-model uplift model.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training outcome
        T_train (pandas.Series): Training treatment
        n_estimators (int): Number of estimators
        
    Returns:
        sklift.models.TwoModels: Trained uplift model
    """
    # Initialize models
    treatment_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    control_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    # Create and fit two-model uplift model
    uplift_model = TwoModels(
        estimator_trmnt=treatment_model,
        estimator_ctrl=control_model,
        method='vanilla'
    )
    
    uplift_model.fit(X_train, treatment=T_train, y=y_train)
    
    return uplift_model


def evaluate_uplift_model(model, X_test, y_test, T_test):
    """
    Evaluate an uplift model.
    
    Args:
        model (sklift.models.TwoModels): Trained uplift model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test outcome
        T_test (pandas.Series): Test treatment
        
    Returns:
        tuple: (uplift_predictions, auc_score, average_uplift)
    """
    # Generate uplift predictions
    uplift_preds = model.predict(X_test)
    
    # Calculate AUC score
    auc_score = roc_auc_score(y_test, uplift_preds)
    
    # Calculate average uplift effect
    average_uplift = uplift_preds.mean()
    
    return uplift_preds, auc_score, average_uplift


def plot_uplift_curve(y_test, uplift_preds, T_test):
    """
    Plot an uplift curve.
    
    Args:
        y_test (pandas.Series): Test outcome
        uplift_preds (numpy.ndarray): Uplift predictions
        T_test (pandas.Series): Test treatment
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Compute uplift curve
    uplift_values = uplift_curve(y_test, uplift_preds, T_test)
    
    # Plot uplift curve
    plt.figure(figsize=(8, 6))
    plt.plot(uplift_values, marker='o', linestyle='-', color='blue')
    plt.axhline(y=0, color='gray', linestyle='--', label="Baseline")
    plt.xlabel("Quantiles")
    plt.title("Uplift Curve")
    plt.ylabel("Uplift Score")
    plt.legend()
    plt.grid()
    
    return plt.gcf()


def create_brand_encoding(df, brand_id_col='brand_id', price_col='price_usd'):
    """
    Create brand encoding based on average price.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        brand_id_col (str): Brand ID column
        price_col (str): Price column
        
    Returns:
        pandas.DataFrame: Dataframe with brand encoding
    """
    df_result = df.copy()
    
    # Encode brands by their average price
    df_result['brand_encoded'] = df_result.groupby(brand_id_col)[price_col].transform('mean')
    
    return df_result


def analyze_price_elasticity(df):
    """
    Analyze price elasticity using DoubleML.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: (model, coefficient, standard_error)
    """
    print("\n--- Analyzing Price Elasticity with DoubleML ---\n")
    
    # Define variables
    Y = 'sold'
    T = 'price_usd'
    X = get_control_variables()
    
    # Ensure control variables exist
    X = ensure_control_variables(df, X)
    
    # Prepare data
    data = prepare_doubleml_data(df, Y, T, X)
    
    # Train model
    model, coef, se = train_doubleml_model(data)
    
    # Print results
    print(f"Estimated Price Elasticity: {coef}")
    print(f"Standard Errors: {se}")
    print(interpret_elasticity(coef))
    
    # Plot treatment effects
    treatment_effects = model.predictions["ml_l"]
    plot_treatment_effects(df, T, treatment_effects, 'Uplift Plot: Treatment Effect vs Price (USD)')
    plt.show()
    
    # Plot alternative treatment effects
    treatment_effects_alt = model.predictions["ml_m"]
    plot_treatment_effects(df, T, treatment_effects_alt, 'Alternative Uplift Plot: Treatment Effect vs Price (USD)')
    plt.show()
    
    return model, coef, se


def analyze_seller_price_elasticity(df):
    """
    Analyze seller price elasticity using DoubleML.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: (model, coefficient, standard_error)
    """
    print("\n--- Analyzing Seller Price Elasticity with DoubleML ---\n")
    
    # Define variables
    Y = 'sold'
    T = 'seller_price'
    X = get_control_variables()
    
    # Ensure control variables exist
    X = ensure_control_variables(df, X)
    
    # Prepare data
    data = prepare_doubleml_data(df, Y, T, X)
    
    # Train model
    model, coef, se = train_doubleml_model(data)
    
    # Print results
    print(f"Estimated Seller Price Elasticity: {coef}")
    print(f"Standard Errors: {se}")
    print(interpret_elasticity(coef))
    
    return model, coef, se


def analyze_with_uplift_model(df):
    """
    Analyze price elasticity using uplift modeling.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: (model, auc_score, average_uplift)
    """
    print("\n--- Analyzing Price Elasticity with Uplift Modeling ---\n")
    
    # Create price categories
    df_categories = create_price_categories(df)
    
    # Define features
    X = get_control_variables()
    X = ensure_control_variables(df, X)
    
    # Prepare data
    X_train, X_test, y_train, y_test, T_train, T_test = prepare_uplift_data(df_categories, X)
    
    # Train uplift model
    uplift_model = train_uplift_model(X_train, y_train, T_train)
    
    # Evaluate model
    uplift_preds, auc_score, average_uplift = evaluate_uplift_model(uplift_model, X_test, y_test, T_test)
    
    # Print results
    print(f"ROC AUC Score: {auc_score}")
    print(f"Average Uplift Effect: {average_uplift}")
    
    if average_uplift < 0:
        print("Demand decreases as price increases (Elastic).")
    else:
        print("Demand increases with price (Inelastic).")
    
    # Plot uplift curve
    plot_uplift_curve(y_test, uplift_preds, T_test)
    plt.show()
    
    return uplift_model, auc_score, average_uplift


def analyze_brand_effect(df):
    """
    Analyze brand effect on demand using DoubleML.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: (model, coefficient, standard_error)
    """
    print("\n--- Analyzing Brand Effect with DoubleML ---\n")
    
    # Encode brands
    df_encoded = create_brand_encoding(df)
    
    # Define variables
    Y = 'sold'
    T = 'brand_encoded'
    X = get_control_variables()
    
    # Ensure control variables exist
    X = ensure_control_variables(df, X)
    
    # Prepare data
    data = prepare_doubleml_data(df_encoded, Y, T, X)
    
    # Train model
    model, coef, se = train_doubleml_model(data)
    
    # Print results
    print(f"Estimated Brand Effect: {coef}")
    print(f"Standard Errors: {se}")
    
    if coef < 0:
        print("Demand decreases with the brand type (negative effect).")
    else:
        print("Demand increases with the brand type (positive effect).")
    
    return model, coef, se


def main(df):
    """
    Main function to run all causal analyses.
    
    Args:
        df (pandas.DataFrame): The dataframe from the previous parts
    """
    print("\n===== Starting Price Elasticity Causal Analysis =====\n")
    
    # Price elasticity analysis with DoubleML
    price_model, price_coef, price_se = analyze_price_elasticity(df)
    
    # Seller price elasticity analysis with DoubleML
    seller_model, seller_coef, seller_se = analyze_seller_price_elasticity(df)
    
    # Price elasticity analysis with uplift modeling
    uplift_model, auc_score, avg_uplift = analyze_with_uplift_model(df)
    
    # Brand effect analysis
    brand_model, brand_coef, brand_se = analyze_brand_effect(df)
    
    print("\n===== Causal Analysis Summary =====\n")
    print(f"Price Elasticity (DoubleML): {price_coef:.4f} (SE: {price_se:.4f})")
    print(f"Seller Price Elasticity (DoubleML): {seller_coef:.4f} (SE: {seller_se:.4f})")
    print(f"Uplift Model Performance (AUC): {auc_score:.4f}")
    print(f"Average Uplift Effect: {avg_uplift:.4f}")
    print(f"Brand Effect: {brand_coef:.4f} (SE: {brand_se:.4f})")
    
    print("\nCausal analysis completed successfully!")


if __name__ == "__main__":
    # If this script is run directly, load the data from the previous parts
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