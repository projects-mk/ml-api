from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from otomoto.preprocessing import Preprocessing
from otomoto.model_training import ModelTrainer
from otomoto.utils import get_unique_values
from otomoto.model_predictions import Predictor
from otomoto.input_models import OtomotoInputData
from fastapi import Body
from fastapi import HTTPException

app = FastAPI()


@app.post("/api/v1/otomoto/train_model",tags=['model_training'])
async def train_model():
    try:
        data_processor = Preprocessing()
        data_processor.preprocess_data()

        input_data = data_processor.get_data()

        model_trainer = ModelTrainer(input_data, target_column='price')
        model_trainer.preprocess()
        model_trainer.train_models()

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error occurred: {error}")

    return {"message": "Models trained and registered successfully"}


@app.post("/api/v1/otomoto/get_dropdown_values",tags=['dropdown_values'])
async def get_dropdown_values():
    try:
        unique_values = get_unique_values()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error occurred: {error}")

    return unique_values


@app.post("/api/v1/otomoto/predict",tags=['model_predictions'])
async def predict(data: OtomotoInputData = Body(...)):
    json_data = data.model_dump()
    model_name = json_data['model_name']
    transformer_name = 'otomoto_data_encoder'
    features = {k: v for k, v in json_data.items() if k != 'model_name'}    
    try:
        predictor = Predictor(model_name, transformer_name)
        pred = predictor.predict(features)
    
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error occurred: {error}")
    
    return {"message": pred}
    
    