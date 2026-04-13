# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare Olist E-Commerce Data
# MAGIC
# MAGIC Downloads the Olist dataset, joins orders with payments,
# MAGIC aggregates into a monthly time series, and saves to Delta.

# COMMAND ----------

# MAGIC %pip install kagglehub --quiet
# MAGIC %restart_python

# COMMAND ----------

import os
os.environ["KAGGLE_USERNAME"] = dbutils.secrets.get(scope="shm", key="kaggle_user")
os.environ["KAGGLE_KEY"] = dbutils.secrets.get(scope="shm", key="kaggle_key")

dbutils.widgets.text("catalog", "shm")
dbutils.widgets.text("schema", "ts")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Olist dataset

# COMMAND ----------

import kagglehub

path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
print(f"Downloaded to: {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and join orders with payments

# COMMAND ----------

import pandas as pd

orders_pd = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
payments_pd = pd.read_csv(os.path.join(path, "olist_order_payments_dataset.csv"))

order_values = payments_pd.groupby("order_id")["payment_value"].sum().reset_index()
order_values.columns = ["order_id", "order_value"]

orders_pd = orders_pd.merge(order_values, on="order_id", how="inner")
for col in ["order_purchase_timestamp", "order_delivered_customer_date"]:
    orders_pd[col] = pd.to_datetime(orders_pd[col])

orders = spark.createDataFrame(orders_pd)
print(f"Orders: {orders.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate into monthly time series

# COMMAND ----------

from pyspark.sql import functions as F

monthly_delivered = (
    orders
    .filter(F.col("order_delivered_customer_date").isNotNull())
    .withColumn("month", F.date_trunc("month", F.col("order_delivered_customer_date")))
    .groupBy("month")
    .agg(
        F.sum("order_value").alias("total_delivered_value"),
        F.count("*").alias("delivered_count"),
        F.avg(
            F.datediff(
                F.col("order_delivered_customer_date"),
                F.col("order_purchase_timestamp"),
            )
        ).alias("avg_delivery_days"),
    )
    .orderBy("month")
)

monthly_created = (
    orders
    .withColumn("month", F.date_trunc("month", F.col("order_purchase_timestamp")))
    .groupBy("month")
    .agg(F.count("*").alias("new_orders_created"))
    .orderBy("month")
)

monthly_ts = (
    monthly_delivered
    .join(monthly_created, on="month", how="outer")
    .fillna(0)
    .orderBy("month")
)

display(monthly_ts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta

# COMMAND ----------

monthly_ts.createOrReplaceTempView("monthly_ts_temp")
spark.sql(f"CREATE OR REPLACE TABLE {catalog}.{schema}.olist_monthly_ts AS SELECT * FROM monthly_ts_temp")

print(f"Saved to {catalog}.{schema}.olist_monthly_ts")
