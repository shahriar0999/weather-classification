# Weather Classification — Production ML with Ray

## Quick Start
```bash
pip install -r requirements.txt
python src/pipeline/workflow.py --skip-tuning
```

## Steps
1. `src/data/ingest.py`       — Load & validate
2. `src/data/preprocess.py`   — Clean, encode, scale
3. `src/training/trainer.py`  — XGBoost + MLflow
4. `src/tuning/tune.py`       — Ray Tune
5. `src/serving/serve.py`     — Ray Serve API
6. `src/monitoring/monitor.py`— Drift detection
7. `src/pipeline/workflow.py` — Full DAG

## Predict
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Temperature": 15.0, "Humidity": 85.0,
    "Wind_Speed": 20.0, "Precipitation": 70.0,
    "Cloud_Cover": "overcast", "Atmospheric_Pressure": 1005.0,
    "UV_Index": 2, "Season": "Winter",
    "Visibility_km": 4.0, "Location": "inland"
  }'
```

## Ports
| Service       | Port |
|---------------|------|
| API           | 8080 |
| MLflow UI     | 5000 |
| Ray Dashboard | 8265 |
| Prometheus    | 8000 |
