from pyspark.sql import SparkSession, DataFrame

def create_table(spark: SparkSession, table_name: str):
  spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name}")

# Create a new Databricks Connect session. If this fails,
# check that you have configured Databricks Connect correctly.
# See https://docs.databricks.com/dev-tools/databricks-connect.html.
def get_spark() -> SparkSession:
  try:
    from databricks.connect import DatabricksSession
    return DatabricksSession.builder.serverless(True).getOrCreate()
  except ImportError:
    return SparkSession.builder.getOrCreate()

def main():
  pass

if __name__ == '__main__':
  main()
