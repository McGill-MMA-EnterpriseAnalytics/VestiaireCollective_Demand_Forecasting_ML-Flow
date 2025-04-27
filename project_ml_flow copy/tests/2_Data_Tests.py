
# # 2. Data Tests
#
# Test data quality: no duplicates, correct types.


import seller_analysis_notebook_script as sa
import pandas as pd

def test_no_duplicate_rows():
    df = sa.df
    assert df.duplicated().sum() == 0, "Duplicate rows found."

def test_column_types_expected():
    df = sa.df
    expected_types = {
        'has_cross_border_fees': 'bool',
        'buyers_fees': 'float64',
    }
    for col, expected_dtype in expected_types.items():
        assert str(df[col].dtype) == expected_dtype, f"{col} is not {expected_dtype}"

print("Data tests loaded. Run with pytest or manually.")
