
# 🧵 Demand Forecasting – Vestiaire Collective

Welcome! This project explores demand forecasting in the world of pre-owned luxury fashion. By analyzing seller behavior, item characteristics, and pricing dynamics on **Vestiaire Collective**, we build predictive models to support data-driven decision-making.

---

## 📌 Project Goal

To forecast demand (i.e., the likelihood of a product being sold) and provide actionable insights for optimizing pricing and inventory in the second-hand luxury market.

---

## 🌐 Project Links

- 🔗 Company: [Vestiaire Collective](https://us.vestiairecollective.com)
- 📊 Dataset: [Kaggle – Vestiaire Fashion Dataset](https://www.kaggle.com/datasets/justinpakzad/vestiaire-fashion-dataset)

---

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
