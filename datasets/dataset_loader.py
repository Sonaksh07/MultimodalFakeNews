import os
from pathlib import Path
import pandas as pd
from datasets.demo_data import get_demo_df
from configs.logging_config import logger

class DatasetLoader:
    """Handles dataset loading for FakeNewsNet (PolitiFact / GossipCop) and fallback demo datasets."""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, dataset_name: str = "demo") -> pd.DataFrame:
        """Loads requested dataset or falls back to demo data automatically."""
        dataset_name = dataset_name.lower()
        file_path = self.data_dir / f"{dataset_name}.csv"

        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Successfully loaded dataset '{dataset_name}' with {len(df)} records.")
                return df
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}. Falling back to demo data.")

        logger.info(f"Dataset '{dataset_name}' not found on disk. Initializing demo dataset.")
        return get_demo_df()

    def get_available_datasets(self) -> list:
        """Lists available dataset files in data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))
        names = [f.stem for f in csv_files]
        if "demo" not in names:
            names.append("demo")
        return names
