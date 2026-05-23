"""Tests — run: pytest tests/ -v --cov=src"""

import pytest
import pandas as pd
import numpy as np
import pickle
import yaml
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Temperature": [25.0, -5.0, 35.0, 10.0],
            "Humidity": [80.0, 60.0, 90.0, 45.0],
            "Wind Speed": [15.0, 30.0, 5.0, 20.0],
            "Precipitation (%)": [70.0, 5.0, 50.0, 10.0],
            "Cloud Cover": ["overcast", "clear", "partly cloudy", "cloudy"],
            "Atmospheric Pressure": [1005.0, 1020.0, 1010.0, 1015.0],
            "UV Index": [2, 8, 5, 3],
            "Season": ["Winter", "Summer", "Spring", "Autumn"],
            "Visibility (km)": [4.0, 15.0, 8.0, 10.0],
            "Location": ["inland", "coastal", "mountain", "inland"],
            "Weather Type": ["Rainy", "Sunny", "Cloudy", "Cloudy"],
        }
    )


class TestSchema:
    def test_required_columns(self, sample_df, config):
        required = (
            config["data"]["numeric_features"]
            + config["data"]["categorical_features"]
            + [config["data"]["target_column"]]
        )
        for col in required:
            assert col in sample_df.columns

    def test_no_empty_df(self, sample_df):
        assert len(sample_df) > 0

    def test_humidity_clipped(self, sample_df):
        assert sample_df["Humidity"].clip(0, 100).between(0, 100).all()


class TestModel:
    def test_model_predicts(self, config):
        if not os.path.exists(config["model"]["save_path"]):
            pytest.skip("Model not trained yet")
        with open(config["model"]["save_path"], "rb") as f:
            model = pickle.load(f)
        with open("data/processed/data.pkl", "rb") as f:
            data = pickle.load(f)
        preds = model.predict(data["X_test"][:5])
        assert len(preds) == 5

    def test_accuracy_threshold(self, config):
        if not os.path.exists(config["model"]["save_path"]):
            pytest.skip("Model not trained yet")
        from sklearn.metrics import accuracy_score

        with open(config["model"]["save_path"], "rb") as f:
            model = pickle.load(f)
        with open("data/processed/data.pkl", "rb") as f:
            data = pickle.load(f)
        acc = accuracy_score(data["y_test"], model.predict(data["X_test"]))
        assert acc >= config["monitoring"]["accuracy_threshold"]
