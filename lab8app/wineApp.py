from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import mlflow
import numpy as np

app = FastAPI(
    title="Wine Classifier",
    description="Classify wine into different categories.",
    version="0.1"
)

@app.get('/')
def main():
    return {'message': 'This is a model for classifying wine.'}

class request_body(BaseModel):
    vector: list

@app.on_event('startup')
def load_artifacts():
    global model
    mlflow.set_tracking_uri('https://mlflow-test-run-803841580416.us-west2.run.app')
    logged_model = 'runs:/bad1e37f4cd34f62909052670a688759/better_models'
    model = mlflow.pyfunc.load_model(logged_model)

@app.post('/predict')
def predict(data : request_body):
    X = np.array([data.vector])
    prediction = model.predict(X)
    result = int(prediction[0])
    return {'Prediction': result}