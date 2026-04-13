# Timeseries Research

Time series research, benchmarking, and solution accelerators on Databricks.

## Projects

| Folder | Description |
|--------|-------------|
| `intro/` | Educational intro to time series -- LongHorizon Weather dataset, EDA, classical and ML forecasting |
| `m5/` | M5 forecasting -- Nixtla DL models with Ray Tune, Conv-LSTM seq2seq with Optuna |
| `hierarchical/` | Hierarchical time series feature engineering (Traffic dataset) |
| `automl_aiforecast/` | Databricks AutoML Forecast and `AI_FORECAST` SQL function |
| `canonical_mlflow/` | End-to-end MLflow pipeline: M5 data, LightGBM via mlforecast, UC model registry, serving |
| `chronos_uco_forecast/` | Chronos-2 forecasting with covariates using Olist e-commerce data (order pipeline as UCO analog) |

## Setup

```bash
uv sync
uv sync --extra dev
```

## Deployment

Deploy notebooks as Databricks jobs via bundles:

```bash
databricks bundle deploy --target dev
databricks bundle run
```
