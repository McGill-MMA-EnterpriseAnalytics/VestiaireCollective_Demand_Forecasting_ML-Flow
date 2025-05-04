"""
Test module for Price_Elasticity_2ndPart.py

This module contains unit tests for the functions in the price elasticity modeling module.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import Price_Elasticity_2ndPart as pe2


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'seller_price': [100, 200, 150, 250, 180],
        'seller_badge_encoded': [1, 0, 1, 1, 0],
        'should_be_gone': [0, 1, 0, 0, 1],
        'seller_pass_rate': [0.9, 0.8, 0.95, 0.85, 0.7],
        'price_to_earning_ratio': [1.2, 1.4, 1.1, 1.3, 1.5],
        'seller_products_sold': [100, 50, 150, 200, 75],
        'price_per_like': [20, 25, 15, 30, 18],
        'brand_id': [1, 2, 1, 3, 2],
        'product_type': [1, 2, 1, 3, 2],
        'product_material': [1, 2, 3, 1, 2],
        'product_like_count': [5, 8, 10, 8, 10],
        'seller_num_products_listed': [20, 30, 40, 25, 15],
        'seller_community_rank': [10, 20, 5, 15, 25],
        'seller_activity_ratio': [5, 1.7, 3.8, 8, 5],
        'product_color_encoded': [1, 2, 3, 1, 2],
        'seller_num_followers': [100, 200, 300, 150, 250],
        'margin_rate': [0.1, 0.2, 0.15, 0.12, 0.18],
        'available': [1, 1, 0, 1, 0],
        'seller_country': [1, 2, 1, 3, 2],
        'in_stock': [1, 1, 0, 1, 0],
        'product_season_encoded': [1, 2, 3, 4, 1],
        'usually_ships_within_encoded': [1, 2, 1, 3, 2],
        'product_condition_encoded': [1, 1, 2, 1, 2],
        'warehouse_name_encoded': [1, 2, 3, 1, 2],
        'sold': [1, 0, 0, 1, 0]
    })


def test_get_important_features():
    """Test getting the list of important features."""
    features = pe2.get_important_features()
    
    # Check that the function returns a list
    assert isinstance(features, list)
    
    # Check that the list contains the expected features
    essential_features = ['seller_price', 'seller_badge_encoded', 'product_like_count']
    for feature in essential_features:
        assert feature in features


def test_get_important_features_no_price():
    """Test getting the list of important features excluding seller_price."""
    features = pe2.get_important_features_no_price()
    
    # Check that the function returns a list
    assert isinstance(features, list)
    
    # Check that seller_price is not in the list
    assert 'seller_price' not in features
    
    # Check that other important features are still there
    assert 'seller_badge_encoded' in features
    assert 'product_like_count' in features


def test_prepare_data_sample(sample_df):
    """Test preparing a data sample for model training."""
    # Get important features
    features = pe2.get_important_features()
    
    # Test the function
    X_sample, y_sample, X_train, X_test, y_train, y_test = pe2.prepare_data_sample(
        sample_df, features
    )
    
    # Check that the shapes are correct
    assert X_sample.shape[0] <= sample_df.shape[0]  # Sample should be smaller than original
    assert X_sample.shape[1] == len(features)       # Should have the correct number of features
    assert X_train.shape[0] + X_test.shape[0] == X_sample.shape[0]  # Train + test = sample
    assert len(y_sample) == X_sample.shape[0]       # Target should match features
    assert len(y_train) == X_train.shape[0]         # Train target should match train features
    assert len(y_test) == X_test.shape[0]           # Test target should match test features


def test_prepare_full_data(sample_df):
    """Test preparing the full dataset for model training."""
    # Get important features
    features = pe2.get_important_features()
    
    # Test the function
    X_train, X_test, y_train, y_test = pe2.prepare_full_data(
        sample_df, features
    )
    
    # Check that the shapes are correct
    assert X_train.shape[0] + X_test.shape[0] == sample_df.shape[0]  # Train + test = full data
    assert X_train.shape[1] == len(features)       # Should have the correct number of features
    assert len(y_train) == X_train.shape[0]        # Train target should match train features
    assert len(y_test) == X_test.shape[0]          # Test target should match test features


def test_apply_smote_tomek(sample_df):
    """Test applying SMOTETomek for handling class imbalance."""
    # Prepare data
    features = pe2.get_important_features()
    X_train, X_test, y_train, y_test = pe2.prepare_full_data(
        sample_df, features
    )
    
    # Mock the SMOTETomek class
    with patch('Price_Elasticity_2ndPart.SMOTETomek') as mock_smote:
        # Set up the mock
        mock_instance = MagicMock()
        mock_instance.fit_resample.return_value = (X_train.copy(), pd.Series([1, 0, 1, 0]))
        mock_smote.return_value = mock_instance
        
        # Test the function
        X_resampled, y_resampled = pe2.apply_smote_tomek(X_train, y_train)
        
        # Check that SMOTETomek was called with the correct parameters
        mock_smote.assert_called_once_with(random_state=42)
        mock_instance.fit_resample.assert_called_once()
        
        # Check the output
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)


def test_train_and_evaluate(sample_df):
    """Test training and evaluating a model."""
    # Prepare data
    features = pe2.get_important_features()
    X_train, X_test, y_train, y_test = pe2.prepare_full_data(
        sample_df, features
    )
    
    # Create a simple model for testing
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    
    # Mock the predict_proba method to return a fixed array
    with patch.object(RandomForestClassifier, 'predict_proba', 
                      return_value=np.array([[0.7, 0.3], [0.2, 0.8]])):
        
        # Mock the roc_auc_score function
        with patch('Price_Elasticity_2ndPart.roc_auc_score', return_value=0.75):
            
            # Test the function
            result_model = pe2.train_and_evaluate(model, X_train, y_train, X_test, y_test)
            
            # Check that the model was returned
            assert result_model is model


def test_initialize_models():
    """Test initializing the boosting models."""
    # Test with fast=True
    xgb_model, lgb_model, cat_model = pe2.initialize_models(fast=True)
    
    # Check model parameters
    assert xgb_model.n_estimators == 30
    assert lgb_model.n_estimators == 30
    assert cat_model.get_params()['iterations'] == 30
    
    # Test with fast=False
    xgb_model, lgb_model, cat_model = pe2.initialize_models(fast=False)
    
    # Check model parameters
    assert xgb_model.n_estimators == 100
    assert lgb_model.n_estimators == 100
    assert cat_model.get_params()['iterations'] == 100


def test_plot_learning_curve():
    """Test plotting the learning curve."""
    # Create a simple model for testing
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    
    # Create simple feature and target arrays
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    
    # Mock the learning_curve function
    with patch('Price_Elasticity_2ndPart.learning_curve') as mock_learning_curve:
        # Set up the mock return values
        train_sizes = np.array([0.2, 0.5, 0.8])
        train_scores = np.array([[0.7, 0.8], [0.75, 0.85], [0.8, 0.9]])
        test_scores = np.array([[0.6, 0.7], [0.65, 0.75], [0.7, 0.8]])
        mock_learning_curve.return_value = (train_sizes, train_scores, test_scores)
        
        # Mock the plot function
        with patch('matplotlib.pyplot.plot') as mock_plot:
            # Test the function
            pe2.plot_learning_curve(model, X, y, "TestModel", fast=True)
            
            # Check that the learning_curve function was called
            mock_learning_curve.assert_called_once()
            
            # Check that the plot function was called twice (once for train, once for test)
            assert mock_plot.call_count == 2


def test_setup_plot():
    """Test setting up a plot."""
    # Mock the figure and title functions
    with patch('matplotlib.pyplot.figure') as mock_figure:
        with patch('matplotlib.pyplot.title') as mock_title:
            # Test the function
            pe2.setup_plot("Test Title")
            
            # Check that the figure and title functions were called
            mock_figure.assert_called_once()
            mock_title.assert_called_once_with("Test Title")


def test_finalize_plot():
    """Test finalizing a plot."""
    # Mock the legend, tight_layout, and show functions
    with patch('matplotlib.pyplot.legend') as mock_legend:
        with patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            with patch('matplotlib.pyplot.show') as mock_show:
                # Test the function
                pe2.finalize_plot()
                
                # Check that all functions were called
                mock_legend.assert_called_once()
                mock_tight_layout.assert_called_once()
                mock_show.assert_called_once()


def test_analyze_feature_correlations(sample_df):
    """Test analyzing feature correlations with the target."""
    # Prepare data
    features = pe2.get_important_features()
    X_train = sample_df[features]
    y_train = sample_df['sold']
    
    # Test the function
    correlations = pe2.analyze_feature_correlations(X_train, y_train)
    
    # Check the output
    assert isinstance(correlations, pd.Series)
    assert correlations.shape[0] == len(features)
    assert all([feature in correlations.index for feature in features])


def test_run_fast_comparison(sample_df):
    """Test running a fast comparison of models."""
    # Mock the necessary functions
    with patch('Price_Elasticity_2ndPart.prepare_data_sample') as mock_prepare:
        with patch('Price_Elasticity_2ndPart.initialize_models') as mock_init:
            with patch('Price_Elasticity_2ndPart.train_and_evaluate') as mock_train:
                with patch('Price_Elasticity_2ndPart.setup_plot') as mock_setup:
                    with patch('Price_Elasticity_2ndPart.plot_learning_curve') as mock_plot:
                        with patch('Price_Elasticity_2ndPart.finalize_plot') as mock_finalize:
                            # Mock return values
                            mock_prepare.return_value = (None, None, None, None, None, None)
                            mock_init.return_value = (MagicMock(), MagicMock(), MagicMock())
                            mock_train.return_value = MagicMock()
                            
                            # Test the function
                            pe2.run_fast_comparison(sample_df)
                            
                            # Check that all the necessary functions were called
                            mock_prepare.assert_called_once()
                            mock_init.assert_called_once_with(fast=True)
                            assert mock_train.call_count == 3
                            mock_setup.assert_called_once()
                            assert mock_plot.call_count == 3
                            mock_finalize.assert_called_once()


def test_run_with_smote_tomek(sample_df):
    """Test running models with SMOTETomek."""
    # Mock the necessary functions
    with patch('Price_Elasticity_2ndPart.prepare_full_data') as mock_prepare:
        with patch('Price_Elasticity_2ndPart.apply_smote_tomek') as mock_smote:
            with patch('Price_Elasticity_2ndPart.initialize_models') as mock_init:
                with patch('Price_Elasticity_2ndPart.train_and_evaluate') as mock_train:
                    with patch('Price_Elasticity_2ndPart.setup_plot') as mock_setup:
                        with patch('Price_Elasticity_2ndPart.plot_learning_curve') as mock_plot:
                            with patch('Price_Elasticity_2ndPart.finalize_plot') as mock_finalize:
                                # Mock return values
                                mock_prepare.return_value = (None, None, None, None)
                                mock_smote.return_value = (None, None)
                                mock_init.return_value = (MagicMock(), MagicMock(), MagicMock())
                                mock_train.return_value = MagicMock()
                                
                                # Test the function
                                result = pe2.run_with_smote_tomek(sample_df)
                                
                                # Check that all the necessary functions were called
                                mock_prepare.assert_called_once()
                                mock_smote.assert_called_once()
                                mock_init.assert_called_once_with(fast=False)
                                assert mock_train.call_count == 3
                                mock_setup.assert_called_once()
                                assert mock_plot.call_count == 3
                                mock_finalize.assert_called_once()
                                
                                # Check the result structure
                                assert len(result) == 7


def test_run_without_seller_price(sample_df):
    """Test running models without seller_price."""
    # Mock the necessary functions
    with patch('Price_Elasticity_2ndPart.prepare_full_data') as mock_prepare:
        with patch('Price_Elasticity_2ndPart.apply_smote_tomek') as mock_smote:
            with patch('Price_Elasticity_2ndPart.initialize_models') as mock_init:
                with patch('Price_Elasticity_2ndPart.train_and_evaluate') as mock_train:
                    with patch('Price_Elasticity_2ndPart.setup_plot') as mock_setup:
                        with patch('Price_Elasticity_2ndPart.plot_learning_curve') as mock_plot:
                            with patch('Price_Elasticity_2ndPart.finalize_plot') as mock_finalize:
                                with patch('Price_Elasticity_2ndPart.analyze_feature_correlations') as mock_analyze:
                                    # Mock return values
                                    mock_prepare.return_value = (None, None, None, None)
                                    mock_smote.return_value = (None, None)
                                    mock_init.return_value = (MagicMock(), MagicMock(), MagicMock())
                                    mock_train.return_value = MagicMock()
                                    mock_analyze.return_value = pd.Series()
                                    
                                    # Test the function
                                    pe2.run_without_seller_price(sample_df)
                                    
                                    # Check that all the necessary functions were called
                                    mock_prepare.assert_called_once()
                                    mock_smote.assert_called_once()
                                    mock_init.assert_called_once_with(fast=False)
                                    assert mock_train.call_count == 3
                                    mock_setup.assert_called_once()
                                    assert mock_plot.call_count == 3
                                    mock_finalize.assert_called_once()
                                    mock_analyze.assert_called_once()


def test_main(sample_df):
    """Test the main function."""
    # Mock the component functions
    with patch('Price_Elasticity_2ndPart.run_fast_comparison') as mock_fast:
        with patch('Price_Elasticity_2ndPart.run_with_smote_tomek') as mock_smote:
            with patch('Price_Elasticity_2ndPart.run_without_seller_price') as mock_no_price:
                # Mock return values
                mock_smote.return_value = (None, None, None, None, None, None, None)
                
                # Test the function
                pe2.main(sample_df)
                
                # Check that all component functions were called
                mock_fast.assert_called_once_with(sample_df)
                mock_smote.assert_called_once_with(sample_df)
                mock_no_price.assert_called_once_with(sample_df)