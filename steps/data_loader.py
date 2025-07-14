import logging
import os
import zipfile
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)

@step
def load_data(data_path: str = "C://Users//User//Documents//Portfolio Projects//Python Projects//Delivery-Duration-Prediction-Mlops//datasets.zip") -> pd.DataFrame:
    """Loads Historical_data.csv from a local ZIP archive.

    Args:
        data_path: Path to the zip file.

    Returns:
        Pandas DataFrame with loaded data.
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} does not exist.")

        extract_path = "historic_data"

        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        csv_path = os.path.join(extract_path, "datasets", "Historical_data.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find Historical_data.csv in {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise