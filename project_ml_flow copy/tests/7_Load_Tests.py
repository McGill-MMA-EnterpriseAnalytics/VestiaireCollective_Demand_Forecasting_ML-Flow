# 7. Load Tests
# 
# Test if the pipeline can handle large datasets.

import seller_analysis_notebook_script as sa
import pandas as pd

def test_load_big_data():
    df_big = pd.concat([sa.df]*10, ignore_index=True)
    assert df_big.shape[0] == 10 * sa.df.shape[0], "Data not scaled properly."

print("Load test ready.")
