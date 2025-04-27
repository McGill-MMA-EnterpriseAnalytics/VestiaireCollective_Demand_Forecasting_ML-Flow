
# # 1. Unit Tests
#
# Test small isolated logic blocks: missing value imputation, column dropping, etc.


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("__file__"), '../src')))

import seller_analysis_notebook_script as sa
import pandas as pd

def test_no_missing_values():
    df = sa.df
    assert df.isnull().sum().sum() == 0, "There are missing values remaining."

def test_irrelevant_columns_dropped():
    df = sa.df
    dropped_cols = [
        'product_keywords', 'product_id', 'product_name',
        'product_description', 'brand_id', 'brand_url', 'seller_username'
    ]
    for col in dropped_cols:
        assert col not in df.columns, f"Column {col} was not dropped."

print("Unit tests loaded. Run with pytest or call functions manually.")
