"""
Test module for Price_Elasticity_3rdPart.py

This module contains unit tests for the functions in the price elasticity causal analysis module.
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

import Price_Elasticity_3rdPart as pe3


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'sold': [1, 0, 0, 1, 0, 1, 0, 1],
        'price_usd': [100, 200, 150, 50, 180, 120, 90, 220],
        'seller_price': [90, 180, 130, 45, 160, 100, 80, 200],
        'product_type': [1, 2, 1, 3, 2, 1, 3, 2],
        'product_category': [1, 2, 1, 2, 3, 1, 2, 3],
        'product_season': [1, 2, 3, 1, 2, 4, 1, 3],
        'product_condition': [1, 1, 2, 1, 2, 1, 2, 1],
        'seller_badge': [1, 0, 1, 0, 1, 1, 0, 1],
        'has_cross_border_fees': [0, 1, 0, 1, 0, 1, 0, 1],
        'seller_pass_rate': [0.9, 0.8, 0.7, 0.95, 0.85, 0.9, 0.75, 0.8],
        'price_per_like': [20, 40, 30, 10, 45, 24, 18, 55],
        'price_to_earning_ratio': [1.2, 1.1, 1.3, 1.1, 1.2, 1.3, 1.1, 1.2],
        'seller_activity_ratio': [2.5, 1.5, 3.0, 2.0, 2.2, 1.8, 2.8, 1.9],
        'brand_id': [1, 2, 1, 3, 2, 1, 3, 2],
        'product_like_count': [5, 5, 5, 5, 4, 5, 5, 4]
    })


def test_get_control_variables():
    """Test getting the list of control variables."""
    control_vars = pe3.get_control_variables()
    
    # Check that the function returns a list
    assert isinstance(control_vars, list)
    
    # Check that the list contains expected control variables
    essential_controls = ['product_type', 'seller_badge', 'price_per_like']
    for var in essential_controls:
        assert var in control_vars


def test_ensure_control_variables(sample_df):
    """Test ensuring control variables exist in the dataframe."""
    # Define a list with some variables that exist and some that don't
    control_vars = [
        'product_type',
        'seller_badge',
        'nonexistent_variable',
        'price_per_like'
    ]
    
    # Test the function
    filtered_vars = pe3.ensure_control_variables(sample_df, control_vars)
    
    # Check that nonexistent variables were removed
    assert 'nonexistent_variable' not in filtered_vars
    
    # Check that existing variables were kept
    assert 'product_type' in filtered_vars
    assert 'seller_badge' in filtered_vars
    assert 'price_per_like' in filtered_vars


def test_prepare_doubleml_data(sample_df):
    """Test preparing data for DoubleML."""
    # Define variables
    Y = 'sold'
    T = 'price_usd'
    X = ['product_type', 'seller_badge', 'price_per_like']
    
    # Mock the DoubleMLData class
    with patch('Price_Elasticity_3rdPart.DoubleMLData') as mock_doubleml_data:
        # Test the function
        pe3.prepare_doubleml_data(sample_df, Y, T, X)
        
        # Check that DoubleMLData was called with the correct arguments
        mock_doubleml_data.assert_called_once()
        call_args = mock_doubleml_data.call_args[0]
        assert call_args[1] == Y  # y_col
        assert call_args[2] == T  # d_cols
        assert call_args[3] == X  # x_cols


def test_train_doubleml_model():
    """Test training a DoubleML model."""
    # Create a mock data object
    mock_data = MagicMock()
    
    # Mock the DoubleMLPLR class and its fit method
    with patch('Price_Elasticity_3rdPart.DoubleMLPLR') as mock_doublemlplr:
        # Configure the mock
        mock_model = MagicMock()
        mock_model.coef = -0.5
        mock_model.se = 0.1
        mock_doublemlplr.return_value = mock_model
        
        # Test the function
        model, coef, se = pe3.train_doubleml_model(mock_data)
        
        # Check that DoubleMLPLR was called and fit
        mock_doublemlplr.assert_called_once()
        mock_model.fit.assert_called_once()
        
        # Check the returned values
        assert model == mock_model
        assert coef == -0.5
        assert se == 0.1


def test_interpret_elasticity():
    """Test interpreting elasticity coefficients."""
    # Test with negative coefficient (elastic)
    result_elastic = pe3.interpret_elasticity(-0.5)
    assert "Elastic" in result_elastic
    assert "decreases" in result_elastic
    
    # Test with positive coefficient (inelastic)
    result_inelastic = pe3.interpret_elasticity(0.5)
    assert "Inelastic" in result_inelastic
    assert "increases" in result_inelastic


def test_plot_treatment_effects(sample_df):
    """Test plotting treatment effects."""
    # Create sample treatment effects
    treatment_effects = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, -0.3, 0.2])
    
    # Mock plt.figure and plt.gcf
    with patch('matplotlib.pyplot.figure') as mock_figure:
        with patch('matplotlib.pyplot.gcf') as mock_gcf:
            mock_fig = MagicMock()
            mock_gcf.return_value = mock_fig
            
            # Test the function
            result = pe3.plot_treatment_effects(
                sample_df, 'price_usd', treatment_effects, 'Test Plot'
            )
            
            # Check that figure was created and returned
            mock_figure.assert_called_once()
            assert result == mock_fig


def test_create_price_categories(sample_df):
    """Test creating price categories."""
    # Test the function
    result_df = pe3.create_price_categories(sample_df, price_col='price_usd', n_categories=3)
    
    # Check that the new columns were added
    assert 'price_category' in result_df.columns
    assert 'price_binary' in result_df.columns
    
    # Check that price_category has the right number of unique values
    assert result_df['price_category'].nunique() <= 3
    
    # Check that price_binary is binary
    assert set(result_df['price_binary'].unique()).issubset({0, 1})


def test_prepare_uplift_data(sample_df):
    """Test preparing data for uplift modeling."""
    # First create price categories
    df_categories = pe3.create_price_categories(sample_df)
    
    # Define features
    X = ['product_type', 'seller_badge', 'price_per_like']
    Y = 'sold'
    T = 'price_binary'
    
    # Mock train_test_split
    with patch('Price_Elasticity_3rdPart.train_test_split') as mock_split:
        # Configure mock return value
        mock_split.return_value = (
            pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), pd.Series()
        )
        
        # Test the function
        X_train, X_test, y_train, y_test, T_train, T_test = pe3.prepare_uplift_data(
            df_categories, X, Y, T
        )
        
        # Check that train_test_split was called with correct arguments
        mock_split.assert_called_once()
        
        # Check that binary conversion was applied
        call_kwargs = mock_split.call_args[1]
        assert call_kwargs['test_size'] == 0.2
        assert call_kwargs['random_state'] == 42


def test_train_uplift_model():
    """Test training an uplift model."""
    # Create mock data
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    T_train = pd.Series([0, 1, 1])
    
    # Mock RandomForestClassifier and TwoModels
    with patch('Price_Elasticity_3rdPart.RandomForestClassifier') as mock_rf:
        with patch('Price_Elasticity_3rdPart.TwoModels') as mock_twomodels:
            # Configure mocks
            mock_rf_instance = MagicMock()
            mock_rf.return_value = mock_rf_instance
            
            mock_uplift = MagicMock()
            mock_twomodels.return_value = mock_uplift
            
            # Test the function
            result = pe3.train_uplift_model(X_train, y_train, T_train)
            
            # Check that models were properly initialized
            assert mock_rf.call_count == 2
            mock_twomodels.assert_called_once()
            
            # Check that fit was called with correct arguments
            mock_uplift.fit.assert_called_once_with(X_train, treatment=T_train, y=y_train)
            
            # Check the returned model
            assert result == mock_uplift


def test_evaluate_uplift_model():
    """Test evaluating an uplift model."""
    # Create mock data
    X_test = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    y_test = pd.Series([0, 1])
    T_test = pd.Series([0, 1])
    
    # Create mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.2, 0.7])
    
    # Mock roc_auc_score
    with patch('Price_Elasticity_3rdPart.roc_auc_score', return_value=0.8):
        # Test the function
        uplift_preds, auc_score, average_uplift = pe3.evaluate_uplift_model(
            mock_model, X_test, y_test, T_test
        )
        
        # Check that predict was called
        mock_model.predict.assert_called_once_with(X_test)
        
        # Check the results
        assert np.array_equal(uplift_preds, np.array([0.2, 0.7]))
        assert auc_score == 0.8
        assert average_uplift == 0.45  # (0.2 + 0.7) / 2