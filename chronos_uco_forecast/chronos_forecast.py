# Databricks notebook source
# MAGIC %md
# MAGIC # Chronos-2 Forecasting with Covariates
# MAGIC
# MAGIC Reads the prepared Olist monthly time series and runs Chronos-2
# MAGIC forecasting -- univariate and with exogenous variables.

# COMMAND ----------

# MAGIC %pip install chronos-forecasting matplotlib --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "shm")
dbutils.widgets.text("schema", "ts")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load monthly time series

# COMMAND ----------

import pandas as pd

ts_df = spark.table(f"{catalog}.{schema}.olist_monthly_ts").toPandas()
ts_df["month"] = pd.to_datetime(ts_df["month"])
ts_df = ts_df.sort_values("month").reset_index(drop=True)
ts_df["id"] = "olist"

ts_df = ts_df.iloc[1:-1].reset_index(drop=True)
print(f"{len(ts_df)} months: {ts_df['month'].min():%Y-%m} to {ts_df['month'].max():%Y-%m}")
ts_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chronos-2 univariate forecast

# COMMAND ----------

from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
)

prediction_length = 3

# COMMAND ----------

univariate_pred = pipeline.predict_df(
    ts_df[["id", "month", "total_delivered_value"]],
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="month",
    target="total_delivered_value",
)

univariate_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chronos-2 forecast with covariates
# MAGIC
# MAGIC Exogenous features: new orders created, delivered count, avg delivery days.
# MAGIC These are past-only covariates -- Chronos-2 learns the historical
# MAGIC correlation between pipeline dynamics and the delivered value target.

# COMMAND ----------

covariate_cols = ["new_orders_created", "delivered_count", "avg_delivery_days"]
context_cols = ["id", "month", "total_delivered_value"] + covariate_cols

covariate_pred = pipeline.predict_df(
    ts_df[context_cols],
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="month",
    target="total_delivered_value",
)

covariate_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, pred_df, title in zip(
    axes,
    [univariate_pred, covariate_pred],
    ["Univariate (target only)", "With covariates (orders, delivery speed)"],
):
    ts_df.plot(x="month", y="total_delivered_value", ax=ax, label="historical", color="tab:blue")
    pred_df.plot(x="month", y="predictions", ax=ax, label="forecast", color="tab:orange")
    ax.fill_between(
        pred_df["month"],
        pred_df["0.1"],
        pred_df["0.9"],
        alpha=0.3,
        color="tab:orange",
        label="80% interval",
    )
    ax.set_title(title)
    ax.set_ylabel("Total delivered order value (BRL)")
    ax.legend()

plt.xlabel("Month")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare univariate vs covariate forecasts

# COMMAND ----------

comparison = (
    univariate_pred[["month", "predictions"]]
    .rename(columns={"predictions": "univariate"})
    .merge(
        covariate_pred[["month", "predictions"]].rename(columns={"predictions": "with_covariates"}),
        on="month",
    )
)
comparison["delta_pct"] = (
    (comparison["with_covariates"] - comparison["univariate"])
    / comparison["univariate"]
    * 100
).round(2)
display(comparison)
