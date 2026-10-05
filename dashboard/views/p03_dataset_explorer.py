import streamlit as st
from datasets.dataset_loader import DatasetLoader
from spark.spark_processor import SparkDataProcessor

def render():
    st.title("Dataset Explorer")
    st.caption("Reference tool for browsing the project's benchmark/demo corpus and running the PySpark "
               "preprocessing pipeline. Independent from Live Detection — does not affect any live analysis.")

    loader = DatasetLoader()
    available = loader.get_available_datasets()

    selected_dataset = st.selectbox("Benchmark Dataset",["Demo Benchmark Dataset"])

    if st.button("Load & Preprocess Dataset"):
        with st.spinner("Loading dataset and triggering PySpark preprocessing pipeline..."):
            df = loader.load_dataset(selected_dataset)
            spark_proc = SparkDataProcessor()
            df_cleaned = spark_proc.preprocess_text_batch(df)

            st.success(f"Successfully loaded {len(df_cleaned)} records using PySpark / Data Pipeline.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Articles", len(df_cleaned))
            c2.metric("Real Articles Count", len(df_cleaned[df_cleaned["label"] == 0]))
            c3.metric("Fake Articles Count", len(df_cleaned[df_cleaned["label"] == 1]))

            st.subheader("Data Records Preview")
            st.dataframe(df_cleaned, width='stretch')
