import pandas as pd
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
except ImportError:
    pass

class DataLoader:
    def __init__(self, use_spark=True):
        self.use_spark = use_spark
        if self.use_spark:
            self.spark = SparkSession.builder \
                .appName("CPG_Decision_Agent_DataLoader") \
                .getOrCreate()

    def load_data(self, filepath: str):
        """
        Loads the CPG sales data from a parquet or csv file.
        Supports both Spark DataFrames (for Databricks) and Pandas (for local prototyping).
        """
        is_parquet = filepath.endswith('.parquet')

        if self.use_spark:
            if is_parquet:
                return self.spark.read.parquet(filepath)
            else:
                return self.spark.read.csv(filepath, header=True, inferSchema=True)
        else:
            if is_parquet:
                return pd.read_parquet(filepath)
            else:
                return pd.read_csv(filepath)

    def load_data_from_table(self, table_name: str):
        """
        Loads the data directly from a Databricks catalog table.
        e.g., 'default.cpg_sales_data'
        """
        if self.use_spark:
            return self.spark.table(table_name)
        else:
            raise ValueError("Loading from a Databricks table requires use_spark=True")
