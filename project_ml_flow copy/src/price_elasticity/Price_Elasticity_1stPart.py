"""
Price Elasticity Analysis Module

This module provides functions for analyzing price elasticity and other
factors affecting product sales in an e-commerce dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


def load_data(file_path):
    """
    Load the dataset from a parquet file.
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    return pd.read_parquet(file_path)


def check_missing_values(df):
    """
    Check for missing values in the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.Series: Count of missing values per column
    """
    return df.isnull().sum()


def compute_correlation_matrix(df):
    """
    Compute and return the correlation matrix for numeric columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    return df.corr(numeric_only=True)


def plot_correlation_matrix(corr_matrix, figsize=(12, 8)):
    """
    Visualize the correlation matrix as a heatmap.
    
    Args:
        corr_matrix (pandas.DataFrame): Correlation matrix
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    return plt.gcf()


def calculate_vif(df, exclude_columns=None):
    """
    Calculate Variance Inflation Factor for features to detect multicollinearity.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        exclude_columns (list): Columns to exclude from VIF calculation
        
    Returns:
        pandas.DataFrame: DataFrame with VIF values for each feature
    """
    if exclude_columns is None:
        exclude_columns = []
        
    # Select only numeric features
    X_vif = df.select_dtypes(include=['number'])
    
    # Drop excluded columns
    X_vif = X_vif.drop(columns=exclude_columns, errors='ignore')
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    return vif_data.sort_values(by="VIF", ascending=False)


def create_margin_rate(df):
    """
    Calculate margin rate feature.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new margin_rate column
    """
    df_copy = df.copy()
    df_copy['margin_rate'] = (df_copy['seller_price'] - df_copy['seller_earning']) / df_copy['seller_price']
    return df_copy


def create_cross_border_category(df):
    """
    Create interaction feature for cross border fees and product category.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new cross_border_category column
    """
    df_copy = df.copy()
    df_copy['cross_border_category'] = df_copy['has_cross_border_fees_encoded'] * df_copy['product_category_encoded']
    return df_copy


def create_derived_features(df):
    """
    Create derived features for modeling.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new derived features
    """
    df_copy = df.copy()
    
    # Add 1 to denominators to avoid division by zero
    df_copy['price_to_earning_ratio'] = df_copy['price_usd'] / (df_copy['seller_earning'] + 1)
    df_copy['price_per_like'] = df_copy['price_usd'] / (df_copy['product_like_count'] + 1)
    df_copy['seller_activity_ratio'] = df_copy['seller_products_sold'] / (df_copy['seller_num_products_listed'] + 1)
    
    return df_copy


def train_xgboost_model(df, excluded_cols, target_col='sold', test_size=0.2, random_state=42):
    """
    Train an XGBoost regression model and return feature importances.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        excluded_cols (list): Columns to exclude from modeling
        target_col (str): Target column name
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, feature_importances_df, X_train, X_test, y_train, y_test)
    """
    # Prepare features and target
    X = df.drop(columns=excluded_cols + [target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train model
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=200, 
                      learning_rate=0.1, max_depth=6, random_state=random_state)
    xgb.fit(X_train, y_train)
    
    # Calculate feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': xgb.feature_importances_
    })
    feature_importances = feature_importances[feature_importances['Importance'] > 0]
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    
    return xgb, feature_importances, X_train, X_test, y_train, y_test


def plot_feature_importance(feature_importances):
    """
    Plot feature importance from a trained model.
    
    Args:
        feature_importances (pandas.DataFrame): DataFrame with Feature and Importance columns
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Significant Features for Demand Prediction (XGBoost)')
    return plt.gcf()


def plot_target_distribution(df, target_col='sold'):
    """
    Plot the distribution of the target variable.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        target_col (str): Target column name
        
    Returns:
        tuple: (summary_stats, missing_values_count, matplotlib.figure.Figure)
    """
    # Summary statistics
    summary_stats = df[target_col].describe()
    
    # Check for missing values
    missing_values = df[target_col].isnull().sum()
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df[target_col], bins=30, kde=True, color='purple')
    plt.title(f'Distribution of {target_col} (Histogram)')
    
    return summary_stats, missing_values, plt.gcf()


def main():
    """
    Main function to execute the price elasticity analysis pipeline.
    """
    # Load data
    df = load_data('/Users/berlybrigith/Downloads/WINTER SEM/WINTER 2/IS Production 2/PROJECT/Folder structure for unit testing/data')
    
    # Check missing values
    missing_values = check_missing_values(df)
    print("Missing values in dataset:")
    print(missing_values)
    
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(df)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Plot correlation matrix
    plot_correlation_matrix(corr_matrix)
    plt.show()
    
    # Calculate VIF
    vif_data = calculate_vif(df, exclude_columns=['price_usd', 'buyers_fees'])
    print("\nVariance Inflation Factors:")
    print(vif_data)
    
    # Create margin rate
    df = create_margin_rate(df)
    
    # Drop unnecessary columns
    df = df.drop(columns=['product_color', 'product_id'])
    
    # Calculate VIF again
    vif_data = calculate_vif(df, exclude_columns=['price_usd', 'buyers_fees'])
    print("\nUpdated VIF values:")
    print(vif_data)
    
    # Create cross-border category
    df = create_cross_border_category(df)
    
    # Calculate VIF again
    vif_data = calculate_vif(df, exclude_columns=['price_usd', 'buyers_fees'])
    print("\nVIF values after adding cross-border category:")
    print(vif_data)
    
    # Drop high-VIF columns
    df = df.drop(columns=['cross_border_category', 'product_category_encoded', 'has_cross_border_fees_encoded'])
    
    # Create derived features
    df = create_derived_features(df)
    
    # Define columns to exclude from modeling
    excluded_cols = ['product_name', 'product_description', 'product_keywords', 
                     'brand_name', 'brand_url', 'seller_username']
    
    # Train XGBoost model
    model, feature_importances, X_train, X_test, y_train, y_test = train_xgboost_model(df, excluded_cols)
    
    # Print feature importances
    print("\nFeature Importances:")
    print(feature_importances)
    
    # Plot feature importances
    plot_feature_importance(feature_importances)
    plt.show()
    
    # Plot target distribution
    summary_stats, missing_count, fig = plot_target_distribution(df)
    print("\nTarget Variable Summary Statistics:")
    print(summary_stats)
    print(f"\nMissing values in target: {missing_count}")
    plt.show()
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()