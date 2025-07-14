import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from typing import Tuple, Annotated
from zenml import step

logger = logging.getLogger(__name__)

@step
def train_prep_model_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    Annotated[np.ndarray, "prep_duration_predictions"],
    Annotated[np.ndarray, "driving_durations"],
    Annotated[np.ndarray, "order_place_durations"],
    Annotated[np.ndarray, "actual_total_durations"]
]:
    """
    Train a model to predict preparation time, then combine it with driving and ordering time.

    Returns:
        prep_duration_predictions: model predictions
        driving_durations: from X_test
        order_place_durations: from X_test
        actual_total_durations: from y_test
    """
    prep_time = y_train - X_train["estimated_store_to_consumer_driving_duration"] - X_train["estimated_order_place_duration"]
    prep_time = prep_time.clip(lower=0)

    X_train_model = X_train.drop(columns=["estimated_store_to_consumer_driving_duration", "estimated_order_place_duration"])
    X_test_model = X_test.drop(columns=["estimated_store_to_consumer_driving_duration", "estimated_order_place_duration"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_model)
    X_test_scaled = scaler.transform(X_test_model)

    model = LGBMRegressor(random_state=42)
    model.fit(X_train_scaled, prep_time)

    prep_time_pred = model.predict(X_test_scaled)

    driving_durations = X_test["estimated_store_to_consumer_driving_duration"].values
    order_place_durations = X_test["estimated_order_place_duration"].values
    actual_total_durations = y_test.values

    logger.info(f"Training prep model on shape: {X_train_scaled.shape}")
    logger.info(f"Test set size: {X_test_scaled.shape}")
    logger.info(f"Prep time predictions shape: {prep_time_pred.shape}")

    return (
        prep_time_pred,
        driving_durations,
        order_place_durations,
        actual_total_durations,
    )