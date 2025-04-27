

# # 6. Data Skew Tests
# 
# Detect significant shifts in data distributions.



import seller_analysis_notebook_script as sa
import pandas as pd

def test_data_skew():
    original_mean = sa.df['buyers_fees'].mean()
    new_data = pd.DataFrame({'buyers_fees': [2.0, 2.5, 3.0]})
    new_mean = new_data['buyers_fees'].mean()
    assert abs(original_mean - new_mean) < 5.0, "Buyers fees distribution shifted significantly."

print("Data skew test ready.")
