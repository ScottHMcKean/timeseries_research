from dbtsr.main import create_table, get_spark

def test_spark():
    spark = get_spark()
    spark.sql("SELECT 1.")
