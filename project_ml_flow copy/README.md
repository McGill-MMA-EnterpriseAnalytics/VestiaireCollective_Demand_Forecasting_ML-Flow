# Demand-Forecasting-Vestiaire
Welcome! If you're intrigued by the evolving world of pre-owned luxury fashion, you're in the right place.

This project delves into Vestiaire Collective, a leading global platform for buying and selling second-hand designer fashion. By analyzing market trends, consumer behavior, seller dynamics, and price elasticity, we aim to uncover actionable insights that drive data-informed strategies for business growth and success.

Join us on this journey to explore the future of sustainable luxury fashion! 🌍👗


Link to company : https://us.vestiairecollective.com  
Link to dataset : https://www.kaggle.com/datasets/justinpakzad/vestiaire-fashion-dataset


## 📁 Project Structure

```text
.
├── auto-ml/                   # AutoML experiments and baselines
├── data/                      # Data ingestion and loading
│   ├── data_loader.py
│   └── __init__.py
├── docs/                      # Documentation and design notes
├── mlruns/                    # MLflow run tracking (auto-generated)
├── models/                    # Exported model artifacts (.pkl, .json, etc.)
├── notebooks/                 # EDA and development notebooks
├── references/                # Research papers, links, external docs
├── reports/                   # Generated reports, visuals, dashboards
├── src/                       # Main application logic and scripts
│   ├── app.py                 # FastAPI app for serving predictions
│   ├── train.py               # Model training pipeline
│   ├── predict.py             # (Optional) Script to run predictions locally
│   ├── config/
│   │   └── best_run_id.txt    # Reference to best MLflow run
│   └── elasticity/            # Price elasticity module
│       ├── Auto_Selection.py
│       ├── Price_Elasticity_1stPart.py
│       ├── Price_Elasticity_2ndPart.py
│       └── __init__.py
├── tests/
│   ├── test_elasticity/
│   └── test_seller_analysis/
├── Dockerfile_predicting      # Dockerfile for FastAPI deployment
├── pyproject.toml             # Poetry dependencies and environment
├── poetry.lock
├── ci_pipeline.yml            # GitHub Actions or CI config
└── README.md

