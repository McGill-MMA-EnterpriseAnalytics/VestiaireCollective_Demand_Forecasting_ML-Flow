# Demand-Forecasting-Vestiaire

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
project_ml_flow copy/
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
│   ├── Auto_Selection.py      # Selecting the best model 
│   ├── config/
│   │   └── best_run_id.txt    # Reference to best MLflow run
│   └── elasticity/            # Price elasticity module
│       ├── Price_Elasticity_1stPart.py
│       ├── Price_Elasticity_2ndPart.py
│       ├── Price_Elasticity_3rdPart.py
│       └── __init__.py
├── tests/
│   ├── test_elasticity/
│   └── test_seller_analysis/
├── Dockerfile_predicting      # Dockerfile for FastAPI deployment
├── pyproject.toml             # Poetry dependencies and environment
├── poetry.lock
├── ci_pipeline.yml            # GitHub Actions or CI config
└── README.md
```

## Dependency Management

This project uses [Poetry](https://python-poetry.org/) for dependency and environment management.

To install all required packages (based on the `pyproject.toml` and locked versions in `poetry.lock`), run:

(bash)
poetry install

This ensures full reproducibility of the development environment.

To activate the virtual environment:

(bash)
poetry shell

---
## Containerized Model Deployment (FastAPI + Docker)
The trained model is served via a FastAPI app containerized with Docker.

1. Build the Docker Image: (bash) docker build -f Dockerfile_predicting -t vestiaire_predict .
   
2. Run the Container: (bash) docker run -p 8000:8000 vestiaire_predict
   
3. Access the Swagger UI: http://localhost:8000/docs
  
This will load the Swagger interface where you can interact with the /predict endpoint to test predictions.
