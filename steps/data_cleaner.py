import logging
import pandas as pd
import numpy as np
from zenml import step

logger = logging.getLogger(__name__)

@step
def data_cleaning_step(df: pd.DataFrame) -> pd.DataFrame:
    """Clean delivery dataset: handle missing values, invalid entries, and create target column.

    Args:
        df: Raw input DataFrame

    Returns:
        Cleaned DataFrame
    """
    required_columns = [
        "created_at", "actual_delivery_time", 
        "subtotal", "min_item_price", "max_item_price", 
        "total_onshift_dashers", "total_busy_dashers", 
        "total_outstanding_orders"
    ]
    df = df.dropna(subset=required_columns)

    numeric_cols = ["subtotal", "min_item_price", "max_item_price",
                    "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]
    for col in numeric_cols:
        df = df[df[col] > 0]

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"], errors="coerce")
    df = df.dropna(subset=["created_at", "actual_delivery_time"])

    df["actual_total_delivery_duration"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds()

    df = df[df["actual_total_delivery_duration"] > 0]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    return df