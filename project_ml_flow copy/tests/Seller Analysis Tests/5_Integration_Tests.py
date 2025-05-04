 # 5. Integration Tests
# 
# Test if the full cleaning pipeline runs successfully.



import seller_analysis_notebook_script as sa

def test_full_pipeline_runs():
    df = sa.df
    assert df.shape[0] > 0, "DataFrame is empty."
    assert isinstance(sa.num_col, list)
    assert isinstance(sa.cat_col, list)
    assert isinstance(sa.bol_col, list)

print("Integration test ready.")
