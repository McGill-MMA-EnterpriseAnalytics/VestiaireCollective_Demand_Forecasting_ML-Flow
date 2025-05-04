import mlflow
from fastapi import FastAPI
from pydantic import BaseModel



# Load the best model
mlflow.set_tracking_uri("file:/app/mlruns")
# Load best run id
with open("best_run_id.txt", "r") as f:
    best_run_id = f.read().strip()

model = mlflow.pyfunc.load_model(
    model_uri=f"runs:/{best_run_id}/model"
)

# FastAPI app
app = FastAPI()

# Health check route
@app.get("/ping")
def ping():
    return {"status": "ok"}

# Input schema
class InputData(BaseModel):
    features: list

# Prediction route
@app.post("/predict")
def predict(input_data: InputData):
    preds = model.predict([input_data.features])
    return {"prediction": preds.tolist()}
