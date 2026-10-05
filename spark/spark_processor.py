import os
import pandas as pd
from configs.logging_config import logger

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, lower, regexp_replace, length, split, size
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning("PySpark not installed or Java unavailable. Operating in Pandas fallback mode.")

class SparkDataProcessor:
    """Big Data preprocessing & batch prediction engine using Apache Spark / PySpark."""
    
    def __init__(self, app_name: str = "FakeNewsBigDataProcessor"):
        self.app_name = app_name
        self.spark = None
        if SPARK_AVAILABLE:
            try:
                self.spark = (
                    SparkSession.builder
                    .appName(self.app_name)
                    .config("spark.driver.memory", "2g")
                    .config("spark.sql.shuffle.partitions", "4")
                    .getOrCreate()
                )
                logger.info("Apache Spark Session initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize PySpark session: {e}. Switching to Pandas mode.")
                self.spark = None

    def preprocess_text_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distributed text cleaning and basic length feature computation."""
        if self.spark is not None:
            try:
                spark_df = self.spark.createDataFrame(df)
                cleaned_df = (
                    spark_df
                    .withColumn("cleaned_text", lower(regexp_replace(col("text"), r"[^a-zA-Z0-9\s]", "")))
                    .withColumn("word_count", size(split(col("cleaned_text"), r"\s+")))
                    .withColumn("char_count", length(col("text")))
                )
                processed_pandas = cleaned_df.toPandas()
                logger.info(f"PySpark successfully preprocessed {len(processed_pandas)} records.")
                return processed_pandas
            except Exception as e:
                logger.error(f"PySpark batch processing failed: {e}. Falling back to Pandas.")

        # Pandas Fallback
        res = df.copy()
        res["cleaned_text"] = res["text"].str.lower().str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        res["word_count"] = res["cleaned_text"].str.split().str.len()
        res["char_count"] = res["text"].str.len()
        return res

    def compute_graph_degree_batch(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Processes interaction edge lists to compute node degree statistics in parallel."""
        if self.spark is not None:
            try:
                spark_edges = self.spark.createDataFrame(edges_df)
                out_degree = spark_edges.groupBy("source").count().withColumnRenamed("count", "out_degree").withColumnRenamed("source", "user_id")
                in_degree = spark_edges.groupBy("target").count().withColumnRenamed("count", "in_degree").withColumnRenamed("target", "user_id")
                
                degrees = out_degree.join(in_degree, on="user_id", how="outer").fillna(0)
                degrees = degrees.withColumn("total_degree", col("out_degree") + col("in_degree"))
                return degrees.toPandas()
            except Exception as e:
                logger.warning(f"PySpark graph degree computation failed: {e}. Falling back to Pandas.")

        # Pandas Fallback
        sources = edges_df["source"].value_counts().rename("out_degree")
        targets = edges_df["target"].value_counts().rename("in_degree")
        deg_df = pd.concat([sources, targets], axis=1).fillna(0).reset_index()
        deg_df.columns = ["user_id", "out_degree", "in_degree"]
        deg_df["total_degree"] = deg_df["out_degree"] + deg_df["in_degree"]
        return deg_df

    def stop(self):
        """Stops Spark session."""
        if self.spark is not None:
            self.spark.stop()
            logger.info("Spark session closed.")
