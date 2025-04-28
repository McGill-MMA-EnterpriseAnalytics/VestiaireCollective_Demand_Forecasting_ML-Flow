# data_loader.py

import gdown
import pandas as pd
import os

def load_data():
    file_id = "1H3u6xOIiJIlpQvPtK4-Hm5Erc4ycUlMq"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "data/cleaned_data.parquet"
    os.makedirs("../data", exist_ok=True)

    if not os.path.exists(output):
        print("Downloading latest dataset...")
        gdown.download(url, output, quiet=False)

    df = pd.read_parquet(output)
    print("Data loaded successfully!")
    return df
if __name__ == "__main__":
    df = load_data()
    print("Loaded data shape:", df.shape)
