# Databricks notebook source
# MAGIC %md
# MAGIC # Canonical MLFLow Example Time Series
# MAGIC This notebook provides a simple boilerplate for using Databricks to do time series machine learning. It includes:
# MAGIC
# MAGIC 1. Load the M5 dataset
# MAGIC 2. Add features using feature engineering
# MAGIC 3. Train a model using MLFLow Tracking
# MAGIC 4. Log the model using MLFLow and Unity Catalog
# MAGIC 5. Load the model and inference
# MAGIC 6. Serve the model using model serving

# COMMAND ----------

!pip install -r ../requirements-db.txt --quiet
%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess

# COMMAND ----------

# MAGIC %md
# MAGIC We load the M5 dataset from Nixtla here

# COMMAND ----------

from datasetsforecast.m5 import M5
m5_dataset = M5(source_url='https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip')
sales, calendar, heirarchy = m5_dataset.load(directory='/Volumes/shm/timeseries/data')

# COMMAND ----------

# MAGIC %md
# MAGIC Sample 100 time series and merge with heirarchy data to give some categorical features

# COMMAND ----------

unique_ids = heirarchy['unique_id'].sample(100).to_list()
data = sales[sales.unique_id.isin(unique_ids)].merge(heirarchy[['unique_id','cat_id','state_id']], on='unique_id', how='inner')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurize
# MAGIC We use Nixtla to accelerate our feature engineering, but can add the feature engineering client in here later.

# COMMAND ----------


from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

test_size = 14

models={
        'lightgbm': lgb.LGBMRegressor(verbosity=-1)
    }

# Let's create lag features
fcst = MLForecast(
    models=models,
    freq='d',
    lags = [1, 2, 3, 7, 14, 30, 360],
    lag_transforms={
        7: [ExpandingMean()],
        14: [RollingMean(window_size=7)],
        30: [RollingMean(window_size=14)],
    }   
)
features_w_label = fcst.preprocess(data, static_features=[])
display(features_w_label.head(5))


# COMMAND ----------

# MAGIC %md
# MAGIC # Train
# MAGIC We now enter our training module but use a lightgbm model to showcase how we use MLFLow. We also use MLFlow to log the model into Unity Catalog. 

# COMMAND ----------

import mlflow
import numpy as np
mlflow.lightgbm.autolog()

with mlflow.start_run():
    train = features_w_label.groupby('unique_id').apply(
        lambda x: x.iloc[:-test_size]
    ).reset_index(drop=True)
    X_train = train.drop(['y','ds'], axis=1)
    y_train = train['y']
    train_data = lgb.Dataset(X_train, label=y_train)
    mlflow.log_table(train, 'train.parquet')

    test = features_w_label.groupby('unique_id').apply(
        lambda x: x.iloc[-test_size:]
    ).reset_index(drop=True)
    X_test = train.drop(['y','ds'], axis=1)
    y_test = train['y']
    test_data = lgb.Dataset(X_test, label=y_test)
    mlflow.log_table(test, 'test.parquet')

    # Set LightGBM parameters
    params = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    mlflow.log_params(params)

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data]
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Log additional metrics if needed
    mlflow.log_metric("custom_metric", np.mean(y_train-y_pred))

# COMMAND ----------

from mlflow.models import infer_signature

# Infer the model signature
signature = infer_signature(X_test, y_pred)

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the model to Unity Catalog with signature
    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path="model",
        registered_model_name="shm.timeseries.lightgbm_model",
        signature=signature
    )

print(f"Model logged successfully in Unity Catalog with run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC In this section we can load a workspace model or one from Unity Catalog and use it in batch inference.

# COMMAND ----------

# Load the logged model directly from mlflow tracking (workspace level)
model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"
loaded_model = mlflow.lightgbm.load_model(model_uri)

# Use the loaded model for predictions
predictions = loaded_model.predict(X_test[:5])
print(predictions)

# COMMAND ----------

# Load the logged model from unity catalog
mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/shm.timeseries.lightgbm_model/1"
loaded_model = mlflow.lightgbm.load_model(model_uri)

# Use the loaded model for predictions
predictions = loaded_model.predict(X_test[:5])
print(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # Serve
# MAGIC
# MAGIC Once we have a model in Unity Catalog, we can also use the deploy client to create an online endpoint.

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client

# Set the registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize the deployment client
client = get_deploy_client("databricks")

# Create the serving endpoint
endpoint = client.create_endpoint(
    name="lightgbm_ts",
    config={
        "served_entities": [
            {
                "name": "lightgbm_ts",
                "entity_name": "shm.timeseries.lightgbm_model",
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": "lightgbm_ts",
                    "traffic_percentage": 100
                }
            ]
        }
    }
)

print(f"Endpoint created: {endpoint}")
