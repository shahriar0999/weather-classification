import ray
import yaml
import time
import pickle
import numpy as np
import logging
import os
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Absolute path of project root
ROOT = os.path.dirname(os.path.abspath(__file__))

class WeatherInput(BaseModel):
    Temperature:          float
    Humidity:             float
    Wind_Speed:           float
    Precipitation:        float
    Cloud_Cover:          str
    Atmospheric_Pressure: float
    UV_Index:             int
    Season:               str
    Visibility_km:        float
    Location:             str

    @field_validator("Humidity")
    @classmethod
    def clamp_humidity(cls, v):
        return max(0.0, min(100.0, v))

    @field_validator("Precipitation")
    @classmethod
    def clamp_precip(cls, v):
        return max(0.0, min(100.0, v))

class WeatherOutput(BaseModel):
    weather_type:  str
    confidence:    float
    probabilities: dict


app = FastAPI(title="Weather API")

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class WeatherClassifier:

    def __init__(self):
        # Use absolute paths so replica finds files regardless of cwd
        config_path = os.path.join(ROOT, "config/config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        def abs(rel):
            return os.path.join(ROOT, rel)

        with open(abs(self.config["model"]["save_path"]), "rb") as f:
            self.model = pickle.load(f)
        with open(abs(self.config["model"]["scaler_path"]), "rb") as f:
            self.scaler = pickle.load(f)
        with open(abs(self.config["model"]["encoder_path"]), "rb") as f:
            self.label_encoders = pickle.load(f)
        with open(abs(self.config["model"]["target_encoder_path"]), "rb") as f:
            self.target_encoder = pickle.load(f)

        self.classes      = list(self.target_encoder.classes_)
        self.numeric_cols = self.config["data"]["numeric_features"]
        self.cat_cols     = self.config["data"]["categorical_features"]
        logger.info(f"✅ Model ready. Classes: {self.classes}")

    @app.get("/health")
    def health(self):
        return {"status": "healthy", "classes": self.classes}

    @app.post("/predict")
    def predict(self, inp: WeatherInput):
        col_map = {
            "Temperature":          "Temperature",
            "Humidity":             "Humidity",
            "Wind_Speed":           "Wind Speed",
            "Precipitation":        "Precipitation (%)",
            "Atmospheric_Pressure": "Atmospheric Pressure",
            "UV_Index":             "UV Index",
            "Visibility_km":        "Visibility (km)",
            "Cloud_Cover":          "Cloud Cover",
            "Season":               "Season",
            "Location":             "Location"
        }
        raw = inp.dict()

        numeric_vals = [
            float(raw[next(k for k, v in col_map.items() if v == c)])
            for c in self.numeric_cols
        ]
        numeric_scaled = self.scaler.transform([numeric_vals])

        cat_vals = []
        for col in self.cat_cols:
            key = next(k for k, v in col_map.items() if v == col)
            le  = self.label_encoders[col]
            val = str(raw[key])
            if val not in le.classes_:
                val = le.classes_[0]
            cat_vals.append(le.transform([val])[0])

        features   = np.concatenate([numeric_scaled[0], cat_vals]).reshape(1, -1)
        pred_class = self.model.predict(features)[0]
        pred_proba = self.model.predict_proba(features)[0]

        return WeatherOutput(
            weather_type  = self.target_encoder.inverse_transform([pred_class])[0],
            confidence    = round(float(pred_proba.max()), 4),
            probabilities = {
                c: round(float(p), 4)
                for c, p in zip(self.classes, pred_proba)
            }
        )


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    serve.start(http_options={"host": "0.0.0.0", "port": 8080})

    serve.run(
        WeatherClassifier.bind(),
        name="weather",
        route_prefix="/"
    )

    logger.info("✅ API is live!")
    logger.info(f"Predict → http://localhost:8080/predict")
    logger.info(f"Health  → http://localhost:8080/health")
    logger.info(f"Docs    → http://localhost:8080/docs")

    while True:
        time.sleep(5)
