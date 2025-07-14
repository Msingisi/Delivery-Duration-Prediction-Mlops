import logging
from typing import Annotated, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step

logger = logging.getLogger(__name__)

@step
def split_data_step(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """Splits the data into train and test sets.

    Args:
        df: Feature-engineered DataFrame with target.
        test_size: Proportion of test data.
        random_state: Random state for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["actual_total_delivery_duration"])
    y = df["actual_total_delivery_duration"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Split completed. X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test