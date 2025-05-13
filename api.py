from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import dill

app = FastAPI()

class InputRecord(BaseModel):
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str
    hit_date: str
    hit_time: float
    hit_number: int
    hit_type: str
    hit_referer: str
    hit_page_path: str
    event_category: str
    event_action: str
    event_label: str
    event_value: float

class Prediction(BaseModel):
    prediction: float

# Загружаем модель
with open('target_action_prediction_model.pkl', 'rb') as file:
    model = dill.load(file)

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(record: InputRecord):
    df = pd.DataFrame([record.dict()])

    preprocessed_df = model['model'].preprocessor.transform(df)
    y_pred = model['model'].model.predict_proba(preprocessed_df)[:, 1]

    return {"prediction": float(y_pred[0])}
