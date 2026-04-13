# Databricks notebook source
# MAGIC %md
# MAGIC # Chronos-2 Forecasting: Zero-Shot vs LoRA Fine-Tuned
# MAGIC
# MAGIC Uses per-seller monthly time series from the Olist dataset. Runs
# MAGIC Chronos-2 zero-shot, then fine-tunes with LoRA and compares. Each
# MAGIC approach is logged as a separate MLflow experiment.

# COMMAND ----------

# MAGIC %pip install chronos-forecasting plotly "peft>=0.13,<1.0" "transformers>=4.45" --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "shm")
dbutils.widgets.text("schema", "ts")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load per-seller monthly time series

# COMMAND ----------

import pandas as pd
import numpy as np

seller_df = spark.table(f"{catalog}.{schema}.olist_seller_monthly_ts").toPandas()
seller_df["month"] = pd.to_datetime(seller_df["month"])
seller_df = seller_df.sort_values(["seller_id", "month"]).reset_index(drop=True)

n_sellers = seller_df["seller_id"].nunique()
print(f"Loaded {len(seller_df):,} rows, {n_sellers} sellers")
seller_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/test split (hold out last 3 months per seller)

# COMMAND ----------

prediction_length = 3
target = "revenue"
covariate_cols = ["order_count", "avg_price", "avg_freight",
                  "avg_delivery_days", "avg_review_score"]

def split_seller(group):
    group = group.sort_values("month")
    n = len(group)
    group["split"] = ["train"] * (n - prediction_length) + ["test"] * prediction_length
    return group

seller_df = seller_df.groupby("seller_id", group_keys=False).apply(split_seller)
train_df = seller_df[seller_df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
test_df = seller_df[seller_df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

print(f"Train: {len(train_df):,} rows ({train_df['seller_id'].nunique()} sellers)")
print(f"Test:  {len(test_df):,} rows ({test_df['seller_id'].nunique()} sellers)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Chronos-2

# COMMAND ----------

from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Zero-shot predictions (all sellers)

# COMMAND ----------

id_col = "seller_id"
ts_col = "month"

zs_univariate_pred = pipeline.predict_df(
    train_df[[id_col, ts_col, target]],
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_col,
    timestamp_column=ts_col,
    target=target,
)

context_cols = [id_col, ts_col, target] + covariate_cols
zs_covariate_pred = pipeline.predict_df(
    train_df[context_cols],
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_col,
    timestamp_column=ts_col,
    target=target,
)

print(f"Zero-shot predictions: {len(zs_univariate_pred):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics helper

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def calc_metrics(test_df, pred_df, target, id_col, ts_col):
    merged = test_df[[id_col, ts_col, target]].merge(
        pred_df[[id_col, ts_col, "predictions"]],
        on=[id_col, ts_col], how="inner",
    )
    merged[target] = merged[target].astype(float)
    merged["predictions"] = merged["predictions"].astype(float)
    actuals = merged[target].values
    preds = merged["predictions"].values
    mask = actuals > 0
    return {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": np.sqrt(mean_squared_error(actuals, preds)),
        "mape": mean_absolute_percentage_error(actuals[mask], preds[mask]) if mask.sum() > 0 else float("nan"),
        "n_series": merged[id_col].nunique(),
        "n_predictions": len(merged),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log zero-shot experiment

# COMMAND ----------

import mlflow

zs_uni_metrics = calc_metrics(test_df, zs_univariate_pred, target, id_col, ts_col)
zs_cov_metrics = calc_metrics(test_df, zs_covariate_pred, target, id_col, ts_col)

mlflow.set_experiment(f"/Users/scott.mckean@databricks.com/chronos_olist_zero_shot")

with mlflow.start_run(run_name="zero_shot") as run:
    mlflow.log_params({
        "model": "amazon/chronos-2",
        "mode": "zero_shot",
        "prediction_length": prediction_length,
        "n_sellers": n_sellers,
        "target": target,
        "covariates": ", ".join(covariate_cols),
    })
    for prefix, m in [("univariate", zs_uni_metrics), ("covariate", zs_cov_metrics)]:
        for k, v in m.items():
            mlflow.log_metric(f"{prefix}_{k}", v)

print("Zero-shot univariate:", {k: round(v, 4) for k, v in zs_uni_metrics.items()})
print("Zero-shot covariate: ", {k: round(v, 4) for k, v in zs_cov_metrics.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## LoRA fine-tuning

# COMMAND ----------

ft_pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
)

train_records = []
for sid, grp in train_df.groupby(id_col):
    grp = grp.sort_values(ts_col)
    record = {"target": grp[target].values.astype(np.float32)}
    record["past_covariates"] = {
        col: grp[col].values.astype(np.float32) for col in covariate_cols
    }
    train_records.append(record)

print(f"Fine-tuning on {len(train_records)} seller time series")

ft_pipeline.fit(
    inputs=train_records,
    prediction_length=prediction_length,
    finetune_mode="lora",
    num_steps=5000,
    learning_rate=1e-5,
    batch_size=32,
)

print("LoRA fine-tuning complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tuned predictions

# COMMAND ----------

ft_univariate_pred = ft_pipeline.predict_df(
    train_df[[id_col, ts_col, target]],
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_col,
    timestamp_column=ts_col,
    target=target,
)

ft_covariate_pred = ft_pipeline.predict_df(
    train_df[context_cols],
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_col,
    timestamp_column=ts_col,
    target=target,
)

print(f"Fine-tuned predictions: {len(ft_univariate_pred):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log fine-tuned experiment

# COMMAND ----------

ft_uni_metrics = calc_metrics(test_df, ft_univariate_pred, target, id_col, ts_col)
ft_cov_metrics = calc_metrics(test_df, ft_covariate_pred, target, id_col, ts_col)

mlflow.set_experiment(f"/Users/scott.mckean@databricks.com/chronos_olist_lora_finetuned")

with mlflow.start_run(run_name="lora_finetuned") as ft_run:
    mlflow.log_params({
        "model": "amazon/chronos-2",
        "mode": "lora_finetuned",
        "prediction_length": prediction_length,
        "n_sellers": n_sellers,
        "target": target,
        "covariates": ", ".join(covariate_cols),
        "lora_steps": 5000,
        "lora_lr": 1e-5,
        "lora_batch_size": 32,
    })
    for prefix, m in [("univariate", ft_uni_metrics), ("covariate", ft_cov_metrics)]:
        for k, v in m.items():
            mlflow.log_metric(f"{prefix}_{k}", v)

print("Fine-tuned univariate:", {k: round(v, 4) for k, v in ft_uni_metrics.items()})
print("Fine-tuned covariate: ", {k: round(v, 4) for k, v in ft_cov_metrics.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison summary

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

metrics_df = pd.DataFrame([
    {"approach": "Zero-shot univariate", **zs_uni_metrics},
    {"approach": "Zero-shot covariate", **zs_cov_metrics},
    {"approach": "LoRA fine-tuned univariate", **ft_uni_metrics},
    {"approach": "LoRA fine-tuned covariate", **ft_cov_metrics},
])
display(metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics bar chart

# COMMAND ----------

metric_names = ["mae", "rmse", "mape"]
fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE", "RMSE", "MAPE"])

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i, metric in enumerate(metric_names, start=1):
    for j, (_, row) in enumerate(metrics_df.iterrows()):
        fig.add_trace(go.Bar(
            x=[row["approach"]],
            y=[row[metric]],
            name=row["approach"] if i == 1 else None,
            marker_color=colors[j],
            showlegend=(i == 1),
            legendgroup=row["approach"],
        ), row=1, col=i)

fig.update_layout(
    height=450, template="plotly_white", barmode="group",
    title="Zero-Shot vs LoRA Fine-Tuned: Regression Metrics",
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actual vs predicted cross plot (test set, sampled sellers)

# COMMAND ----------

sample_sellers = test_df[id_col].unique()[:50]
test_sample = test_df[test_df[id_col].isin(sample_sellers)]

def build_cross_data(test_sample, pred_df, label):
    m = test_sample[[id_col, ts_col, target]].merge(
        pred_df[[id_col, ts_col, "predictions"]],
        on=[id_col, ts_col], how="inner",
    )
    m["approach"] = label
    return m

cross = pd.concat([
    build_cross_data(test_sample, zs_covariate_pred, "Zero-shot"),
    build_cross_data(test_sample, ft_covariate_pred, "LoRA fine-tuned"),
])
cross[target] = cross[target].astype(float)
cross["predictions"] = cross["predictions"].astype(float)

all_vals = pd.concat([cross[target], cross["predictions"]])
axis_min, axis_max = max(0, all_vals.min() * 0.9), all_vals.max() * 1.1

cross_fig = go.Figure()
for approach, color in [("Zero-shot", "#1f77b4"), ("LoRA fine-tuned", "#ff7f0e")]:
    sub = cross[cross["approach"] == approach]
    cross_fig.add_trace(go.Scatter(
        x=sub[target], y=sub["predictions"],
        mode="markers", name=approach,
        marker=dict(size=6, color=color, opacity=0.6),
    ))

cross_fig.add_trace(go.Scatter(
    x=[axis_min, axis_max], y=[axis_min, axis_max],
    mode="lines", name="perfect", line=dict(color="grey", dash="dash"),
))

cross_fig.update_layout(
    title="Actual vs Predicted Revenue (test set, covariate model)",
    xaxis_title="Actual revenue (BRL)",
    yaxis_title="Predicted revenue (BRL)",
    template="plotly_white", height=550,
    xaxis=dict(range=[axis_min, axis_max]),
    yaxis=dict(range=[axis_min, axis_max]),
)
cross_fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample seller forecast visualization

# COMMAND ----------

top_sellers = (
    train_df.groupby(id_col)[target].sum()
    .nlargest(4).index.tolist()
)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"Seller {s[:8]}..." for s in top_sellers],
)

for idx, sid in enumerate(top_sellers):
    row, col = idx // 2 + 1, idx % 2 + 1
    s_train = train_df[train_df[id_col] == sid].sort_values(ts_col)
    s_test = test_df[test_df[id_col] == sid].sort_values(ts_col)

    zs_pred = zs_covariate_pred[zs_covariate_pred[id_col] == sid].sort_values(ts_col)
    ft_pred = ft_covariate_pred[ft_covariate_pred[id_col] == sid].sort_values(ts_col)

    show_legend = idx == 0
    fig.add_trace(go.Scatter(
        x=s_train[ts_col], y=s_train[target],
        mode="lines", name="train", line=dict(color="#1f77b4"),
        legendgroup="train", showlegend=show_legend,
    ), row=row, col=col)

    if len(s_test) > 0:
        fig.add_trace(go.Scatter(
            x=s_test[ts_col], y=s_test[target].astype(float),
            mode="lines+markers", name="test (actual)",
            line=dict(color="#2ca02c", dash="dash"),
            legendgroup="test", showlegend=show_legend,
        ), row=row, col=col)

    if len(zs_pred) > 0:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(zs_pred[ts_col]),
            y=zs_pred["predictions"].astype(float),
            mode="lines+markers", name="zero-shot",
            line=dict(color="#ff7f0e"),
            legendgroup="zs", showlegend=show_legend,
        ), row=row, col=col)

    if len(ft_pred) > 0:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(ft_pred[ts_col]),
            y=ft_pred["predictions"].astype(float),
            mode="lines+markers", name="LoRA fine-tuned",
            line=dict(color="#d62728"),
            legendgroup="ft", showlegend=show_legend,
        ), row=row, col=col)

fig.update_layout(
    height=700, template="plotly_white",
    title="Top 4 Sellers: Zero-Shot vs LoRA Fine-Tuned Forecasts",
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log comparison artifacts

# COMMAND ----------

mlflow.set_experiment(f"/Users/scott.mckean@databricks.com/chronos_olist_lora_finetuned")
with mlflow.start_run(run_id=ft_run.info.run_id):
    mlflow.log_figure(cross_fig, "actual_vs_predicted.html")
    mlflow.log_figure(fig, "top_sellers_forecast.html")
    mlflow.log_table(metrics_df, artifact_file="metrics_comparison.json")
