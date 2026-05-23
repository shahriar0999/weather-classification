"""
STEP 5 — Model Serving (FIXED)
FastAPI app is created INSIDE __init__ to avoid Ray serialization error.
"""

import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import pickle
import numpy as np
import yaml
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class WeatherClassifier:

    def __init__(self):
        # ✅ FastAPI created INSIDE __init__ — not at module level
        self.app = FastAPI(title="Weather Classification API", version="1.0.0")
        self._register_routes()

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

    def _register_routes(self):

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.get("/model-info")
        async def model_info():
            return {
                "classes": self.classes,
                "num_features": len(self.numeric_cols + self.cat_cols),
            }

        @self.app.post("/predict", response_model=WeatherOutput)
        async def predict(inp: WeatherInput):
            return await self._predict(inp)

        @self.app.post("/batch", response_model=BatchOutput)
        async def batch(inp: BatchInput):
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

    async def __call__(self, request):
        return await self.app(request.scope, request.receive, request.send)


def deploy(config: dict):
    ray.init(ignore_reinit_error=True)
    serve.start(
        http_options={
            "host": config["serving"]["host"],
            "port": config["serving"]["port"],
        }
    )
    WeatherClassifier.bind()
    logger.info(f"API  → http://0.0.0.0:{config['serving']['port']}")
    logger.info(f"Docs → http://0.0.0.0:{config['serving']['port']}/docs")


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    deploy(config)
