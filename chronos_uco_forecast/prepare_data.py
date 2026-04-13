# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare Olist E-Commerce Data
# MAGIC
# MAGIC Downloads all 9 Olist dataset files, joins them, and produces two Delta
# MAGIC tables: an aggregate monthly time series and a per-seller monthly time
# MAGIC series (for multi-series forecasting and fine-tuning).

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
# MAGIC ## Download and load all Olist files

# COMMAND ----------

import kagglehub
import pandas as pd

path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
print(f"Downloaded to: {path}")

files = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "categories": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

dfs = {}
for name, fname in files.items():
    fpath = os.path.join(path, fname)
    dfs[name] = pd.read_csv(fpath)
    print(f"{name}: {len(dfs[name]):,} rows, {list(dfs[name].columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join into a single denormalized orders table

# COMMAND ----------

orders = dfs["orders"].copy()
for col in ["order_purchase_timestamp", "order_delivered_customer_date",
            "order_approved_at", "order_estimated_delivery_date"]:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

items = dfs["items"].copy()
items["item_total"] = items["price"] + items["freight_value"]

payment_totals = dfs["payments"].groupby("order_id")["payment_value"].sum().reset_index()
payment_totals.columns = ["order_id", "payment_value"]

review_scores = dfs["reviews"].groupby("order_id")["review_score"].mean().reset_index()

products = dfs["products"][["product_id", "product_category_name"]].copy()
products = products.merge(
    dfs["categories"], on="product_category_name", how="left"
)

enriched = (
    items
    .merge(orders[["order_id", "customer_id", "order_status",
                    "order_purchase_timestamp", "order_delivered_customer_date",
                    "order_estimated_delivery_date"]],
           on="order_id", how="inner")
    .merge(payment_totals, on="order_id", how="left")
    .merge(review_scores, on="order_id", how="left")
    .merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
    .merge(dfs["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left")
    .merge(dfs["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
)

enriched = enriched[enriched["order_status"] == "delivered"].copy()
enriched["delivery_days"] = (
    enriched["order_delivered_customer_date"] - enriched["order_purchase_timestamp"]
).dt.total_seconds() / 86400

print(f"Enriched delivered items: {len(enriched):,}")
enriched.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate: monthly time series (single series)

# COMMAND ----------

enriched["month"] = enriched["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

agg_monthly = (
    enriched.groupby("month")
    .agg(
        total_delivered_value=("item_total", "sum"),
        new_orders_created=("order_id", "nunique"),
        delivered_count=("order_id", "count"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_review_score=("review_score", "mean"),
        n_unique_sellers=("seller_id", "nunique"),
        n_unique_products=("product_id", "nunique"),
        avg_freight=("freight_value", "mean"),
    )
    .reset_index()
    .sort_values("month")
)

print(f"Aggregate monthly: {len(agg_monthly)} months")
display(spark.createDataFrame(agg_monthly))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-seller monthly time series

# COMMAND ----------

seller_monthly_raw = (
    enriched.groupby(["seller_id", "month"])
    .agg(
        revenue=("item_total", "sum"),
        order_count=("order_id", "nunique"),
        item_count=("order_id", "count"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_review_score=("review_score", "mean"),
        n_unique_products=("product_id", "nunique"),
        n_unique_customers=("customer_id", "nunique"),
    )
    .reset_index()
    .sort_values(["seller_id", "month"])
)

all_months = pd.date_range(
    enriched["month"].min(), enriched["month"].max(), freq="MS"
)

def fill_monthly_grid(grp):
    grp = grp.set_index("month").reindex(all_months).rename_axis("month").reset_index()
    grp["seller_id"] = grp["seller_id"].ffill().bfill()
    fill_zero = ["revenue", "order_count", "item_count", "n_unique_products", "n_unique_customers"]
    grp[fill_zero] = grp[fill_zero].fillna(0)
    fill_cols = ["avg_price", "avg_freight", "avg_delivery_days", "avg_review_score"]
    grp[fill_cols] = grp[fill_cols].ffill().bfill()
    return grp

seller_monthly = (
    seller_monthly_raw
    .groupby("seller_id", group_keys=False)
    .apply(fill_monthly_grid)
    .sort_values(["seller_id", "month"])
    .reset_index(drop=True)
)

active_months = seller_monthly_raw.groupby("seller_id")["month"].count()
min_active_months = 8
active_sellers = active_months[active_months >= min_active_months].index
seller_monthly_filtered = seller_monthly[seller_monthly["seller_id"].isin(active_sellers)].copy()

print(f"Total sellers: {enriched['seller_id'].nunique()}")
print(f"Sellers with >= {min_active_months} active months: {len(active_sellers)}")
print(f"Per-seller monthly rows (with gap-fill): {len(seller_monthly_filtered):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save both tables to Delta

# COMMAND ----------

agg_sdf = spark.createDataFrame(agg_monthly)
agg_sdf.createOrReplaceTempView("agg_monthly_temp")
spark.sql(f"CREATE OR REPLACE TABLE {catalog}.{schema}.olist_monthly_ts AS SELECT * FROM agg_monthly_temp")
print(f"Saved {catalog}.{schema}.olist_monthly_ts ({len(agg_monthly)} rows)")

seller_sdf = spark.createDataFrame(seller_monthly_filtered)
seller_sdf.createOrReplaceTempView("seller_monthly_temp")
spark.sql(f"CREATE OR REPLACE TABLE {catalog}.{schema}.olist_seller_monthly_ts AS SELECT * FROM seller_monthly_temp")
print(f"Saved {catalog}.{schema}.olist_seller_monthly_ts ({len(seller_monthly_filtered):,} rows)")
