"""
Test module for price_elasticity.py

This module contains unit tests for the functions in the price_elasticity module.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import Price_Elasticity_1stPart as pe


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'price_usd': [100, 200, 150, 250, 180],
        'buyers_fees': [10, 20, 15, 25, 18],
        'seller_price': [90, 180, 135, 225, 162],
        'seller_earning': [80, 160, 120, 200, 150],
        'product_like_count': [5, 10, 0, 8, 15],
        'seller_products_sold': [20, 50, 30, 100, 10],
        'seller_num_products_listed': [30, 60, 40, 120, 20],
        'product_color': ['red', 'blue', 'green', 'black', 'white'],
        'product_id': [1001, 1002, 1003, 1004, 1005],
        'product_category_encoded': [1, 2, 1, 3, 2],
        'has_cross_border_fees_encoded': [0, 1, 0, 1, 0],
        'sold': [1, 0, 0, 1, 0],
        'product_name': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'],
        'product_description': ['Desc 1', 'Desc 2', 'Desc 3', 'Desc 4', 'Desc 5'],
        'product_keywords': ['kw1', 'kw2', 'kw3', 'kw4', 'kw5'],
        'brand_name': ['Brand 1', 'Brand 2', 'Brand 1', 'Brand 3', 'Brand 2'],
        'brand_url': ['url1', 'url2', 'url1', 'url3', 'url2'],
        'seller_username': ['seller1', 'seller2', 'seller1', 'seller3', 'seller2']
    })


def test_load_data(tmp_path):
    """Test loading data from a parquet file."""
    # Create a temporary DataFrame and save it as parquet
    test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    test_file = tmp_path / "test_data.parquet"
    test_df.to_parquet(test_file)
    
    # Test the function
    loaded_df = pe.load_data(str(test_file))
    
    # Check that the loaded DataFrame is correct
    pd.testing.assert_frame_equal(loaded_df, test_df)


def test_check_missing_values(sample_df):
    """Test checking for missing values."""
    # Create a DataFrame with some missing values
    df_with_missing = sample_df.copy()
    df_with_missing.loc[0, 'price_usd'] = np.nan
    df_with_missing.loc[2, 'product_like_count'] = np.nan
    
    # Test the function
    missing_counts = pe.check_missing_values(df_with_missing)
    
    # Check that the missing counts are correct
    assert missing_counts['price_usd'] == 1
    assert missing_counts['product_like_count'] == 1
    assert missing_counts['seller_price'] == 0


def test_compute_correlation_matrix(sample_df):
    """Test computing the correlation matrix."""
    # Test the function
    corr_matrix = pe.compute_correlation_matrix(sample_df)
    
    # Check that the correlation matrix has the correct shape
    numeric_cols = sample_df.select_dtypes(include=['number']).columns
    assert corr_matrix.shape == (len(numeric_cols), len(numeric_cols))
    
    # Check that the diagonal values are 1.0
    for col in corr_matrix.columns:
        assert corr_matrix.loc[col, col] == 1.0


def test_plot_correlation_matrix():
    """Test plotting the correlation matrix."""
    # Create a simple correlation matrix
    corr_matrix = pd.DataFrame({
        'A': [1.0, 0.5, -0.2],
        'B': [0.5, 1.0, 0.7],
        'C': [-0.2, 0.7, 1.0]
    }, index=['A', 'B', 'C'])
    
    # Test the function with a mock for plt.figure
    with patch('matplotlib.pyplot.figure') as mock_figure:
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        with patch('matplotlib.pyplot.gcf', return_value=mock_fig):
            result = pe.plot_correlation_matrix(corr_matrix)
            assert result == mock_fig


def test_calculate_vif(sample_df):
    """Test calculating VIF values."""
    # Test the function
    vif_data = pe.calculate_vif(sample_df, exclude_columns=['price_usd', 'buyers_fees'])
    
    # Check that the VIF DataFrame has the expected structure
    assert 'Feature' in vif_data.columns
    assert 'VIF' in vif_data.columns
    assert 'price_usd' not in vif_data['Feature'].values
    assert 'buyers_fees' not in vif_data['Feature'].values


def test_create_margin_rate(sample_df):
    """Test creating the margin rate feature."""
    # Test the function
    result_df = pe.create_margin_rate(sample_df)
    
    # Check that the margin_rate column is added and calculated correctly
    assert 'margin_rate' in result_df.columns
    expected_margin_rate = (sample_df['seller_price'] - sample_df['seller_earning']) / sample_df['seller_price']
    pd.testing.assert_series_equal(result_df['margin_rate'], expected_margin_rate)


def test_create_cross_border_category(sample_df):
    """Test creating the cross border category feature."""
    # Test the function
    result_df = pe.create_cross_border_category(sample_df)
    
    # Check that the cross_border_category column is added and calculated correctly
    assert 'cross_border_category' in result_df.columns
    expected_cross_border = sample_df['has_cross_border_fees_encoded'] * sample_df['product_category_encoded']
    pd.testing.assert_series_equal(result_df['cross_border_category'], expected_cross_border)


def test_create_derived_features(sample_df):
    """Test creating derived features."""
    # Test the function
    result_df = pe.create_derived_features(sample_df)
    
    # Check that the derived features are added
    assert 'price_to_earning_ratio' in result_df.columns
    assert 'price_per_like' in result_df.columns
    assert 'seller_activity_ratio' in result_df.columns
    
    # Check calculations for one of the features
    expected_price_per_like = sample_df['price_usd'] / (sample_df['product_like_count'] + 1)
    pd.testing.assert_series_equal(result_df['price_per_like'], expected_price_per_like)


def test_train_xgboost_model(sample_df):
    """Test training the XGBoost model."""
    # Define columns to exclude
    excluded_cols = ['product_name', 'product_description', 'product_keywords', 
                     'brand_name', 'brand_url', 'seller_username']
    
    # Create a mock for XGBRegressor
    with patch('Price_Elasticity_1stPart.XGBRegressor') as mock_xgb:
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.2, 0.3, 0.0, 0.5, 0.0])
        mock_xgb.return_value = mock_model
        
        # Test the function
        model, feature_importances, X_train, X_test, y_train, y_test = pe.train_xgboost_model(
            sample_df, excluded_cols)
        
        # Check that the model was created and fit
        assert mock_xgb.called
        assert mock_model.fit.called
        
        # Check that feature_importances contains only features with importance > 0
        assert len(feature_importances) < len(sample_df.drop(columns=excluded_cols + ['sold']).columns)
        assert all(feature_importances['Importance'] > 0)


def test_plot_feature_importance():
    """Test plotting feature importance."""
    # Create a sample feature importance DataFrame
    feature_importances = pd.DataFrame({
        'Feature': ['A', 'B', 'C'],
        'Importance': [0.5, 0.3, 0.2]
    })
    
    # Test the function with a mock for plt.figure
    with patch('matplotlib.pyplot.figure') as mock_figure:
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        with patch('matplotlib.pyplot.gcf', return_value=mock_fig):
            result = pe.plot_feature_importance(feature_importances)
            assert result == mock_fig


def test_plot_target_distribution(sample_df):
    """Test plotting the target distribution."""
    # Test the function with a mock for plt.figure
    with patch('matplotlib.pyplot.figure') as mock_figure:
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        with patch('matplotlib.pyplot.gcf', return_value=mock_fig):
            summary_stats, missing_count, fig = pe.plot_target_distribution(sample_df)
            
            # Check the results
            assert isinstance(summary_stats, pd.Series)
            assert missing_count == 0
            assert fig == mock_fig


def test_main():
    """Test the main function."""
    # This is more of an integration test, so we'll mock most of the functions
    with patch('Price_Elasticity_1stPart.load_data') as mock_load_data, \
         patch('Price_Elasticity_1stPart.check_missing_values') as mock_check_missing, \
         patch('Price_Elasticity_1stPart.compute_correlation_matrix') as mock_compute_corr, \
         patch('Price_Elasticity_1stPart.plot_correlation_matrix') as mock_plot_corr, \
         patch('Price_Elasticity_1stPart.calculate_vif') as mock_calc_vif, \
         patch('Price_Elasticity_1stPart.create_margin_rate') as mock_create_margin, \
         patch('Price_Elasticity_1stPart.create_cross_border_category') as mock_create_cross, \
         patch('Price_Elasticity_1stPart.create_derived_features') as mock_create_derived, \
         patch('Price_Elasticity_1stPart.train_xgboost_model') as mock_train_xgb, \
         patch('Price_Elasticity_1stPart.plot_feature_importance') as mock_plot_importance, \
         patch('Price_Elasticity_1stPart.plot_target_distribution') as mock_plot_target, \
         patch('matplotlib.pyplot.show') as mock_show:
        
        # Set up the mock returns
        mock_load_data.return_value = pd.DataFrame()
        mock_check_missing.return_value = pd.Series()
        mock_compute_corr.return_value = pd.DataFrame()
        mock_plot_corr.return_value = MagicMock()
        mock_calc_vif.return_value = pd.DataFrame()
        mock_create_margin.return_value = pd.DataFrame()
        mock_create_cross.return_value = pd.DataFrame()
        mock_create_derived.return_value = pd.DataFrame()
        mock_train_xgb.return_value = (MagicMock(), pd.DataFrame(), None, None, None, None)
        mock_plot_importance.return_value = MagicMock()
        mock_plot_target.return_value = (pd.Series(), 0, MagicMock())
        
        # Run the main function
        pe.main()
        
        # Check that all the functions were called
        mock_load_data.assert_called_once()
        mock_check_missing.assert_called_once()
        mock_compute_corr.assert_called_once()
        mock_plot_corr.assert_called_once()
        assert mock_calc_vif.call_count >= 1
        mock_create_margin.assert_called_once()
        mock_create_cross.assert_called_once()
        mock_create_derived.assert_called_once()
        mock_train_xgb.assert_called_once()
        mock_plot_importance.assert_called_once()
        mock_plot_target.assert_called_once()
        assert mock_show.call_count >= 1