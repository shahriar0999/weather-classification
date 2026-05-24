"""STEP 5 — Model Serving"""

import logging
import pickle
import time
from typing import List

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from ray import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fastapi_app = FastAPI(title="Weather Classification API", version="1.0.0")


class WeatherInput(BaseModel):
    Temperature: float = Field(..., example=25.0)
    Humidity: float = Field(..., example=70.0)
    Wind_Speed: float = Field(..., example=15.0)
    Precipitation: float = Field(..., example=40.0)
    Cloud_Cover: str = Field(..., example="partly cloudy")
    Atmospheric_Pressure: float = Field(..., example=1013.0)
    UV_Index: int = Field(..., example=5)
    Season: str = Field(..., example="Winter")
    Visibility_km: float = Field(..., example=10.0)
    Location: str = Field(..., example="inland")

    @validator("Humidity")
    def clamp_humidity(cls, v):
        return max(0.0, min(100.0, v))

    @validator("Precipitation")
    def clamp_precip(cls, v):
        return max(0.0, min(100.0, v))


class WeatherOutput(BaseModel):
    weather_type: str
    confidence: float
    probabilities: dict
    latency_ms: float


class BatchInput(BaseModel):
    records: List[WeatherInput]


class BatchOutput(BaseModel):
    predictions: List[WeatherOutput]
    total_latency_ms: float


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_cpus": 1,
        "runtime_env": {
            "pip": [
                "xgboost",
                "scikit-learn",
                "pydantic",
                "fastapi",
                "pyyaml",
                "numpy",
            ]
        },
    },
)
@serve.ingress(fastapi_app)
class WeatherClassifier:

    def __init__(self):
        with open("config/config.yaml") as f:
            self.config = yaml.safe_load(f)

        with open(self.config["model"]["save_path"], "rb") as f:
            self.model = pickle.load(f)
        with open(self.config["model"]["scaler_path"], "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.config["model"]["encoder_path"], "rb") as f:
            self.label_encoders = pickle.load(f)
        with open(self.config["model"]["target_encoder_path"], "rb") as f:
            self.target_encoder = pickle.load(f)

        self.classes = list(self.target_encoder.classes_)
        self.numeric_cols = self.config["data"]["numeric_features"]
        self.cat_cols = self.config["data"]["categorical_features"]
        logger.info(f"✅ Model loaded. Classes: {self.classes}")

    @fastapi_app.get("/health")
    async def health(self):
        return {"status": "healthy"}

    @fastapi_app.get("/model-info")
    async def model_info(self):
        return {
            "classes": self.classes,
            "num_features": len(self.numeric_cols + self.cat_cols),
        }

    @fastapi_app.post("/predict", response_model=WeatherOutput)
    async def predict(self, inp: WeatherInput):
        return await self._predict(inp)

    @fastapi_app.post("/batch", response_model=BatchOutput)
    async def batch(self, inp: BatchInput):
        t0 = time.time()
        results = [await self._predict(r) for r in inp.records]
        return BatchOutput(
            predictions=results,
            total_latency_ms=round((time.time() - t0) * 1000, 2),
        )

    def _build_features(self, inp: WeatherInput) -> np.ndarray:
        col_map = {
            "Temperature": "Temperature",
            "Humidity": "Humidity",
            "Wind_Speed": "Wind Speed",
            "Precipitation": "Precipitation (%)",
            "Atmospheric_Pressure": "Atmospheric Pressure",
            "UV_Index": "UV Index",
            "Visibility_km": "Visibility (km)",
            "Cloud_Cover": "Cloud Cover",
            "Season": "Season",
            "Location": "Location",
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
            le = self.label_encoders[col]
            val = str(raw[key])
            if val not in le.classes_:
                val = le.classes_[0]
            cat_vals.append(le.transform([val])[0])
        return np.concatenate([numeric_scaled[0], cat_vals]).reshape(1, -1)

    async def _predict(self, inp: WeatherInput) -> WeatherOutput:
        t0 = time.time()
        try:
            features = self._build_features(inp)
            pred_class = self.model.predict(features)[0]
            pred_proba = self.model.predict_proba(features)[0]
            return WeatherOutput(
                weather_type=self.target_encoder.inverse_transform([pred_class])[0],
                confidence=round(float(pred_proba.max()), 4),
                probabilities={
                    c: round(float(p), 4) for c, p in zip(self.classes, pred_proba)
                },
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def deploy(config: dict):
    pass  # deployment now handled by CI/CD


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    deploy(config)


app = WeatherClassifier.bind()
