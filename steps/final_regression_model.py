import logging
import numpy as np
import pandas as pd
from typing import Annotated, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from zenml import step

logger = logging.getLogger(__name__)

@step
def final_regression_model_step(
    prep_predictions: np.ndarray,
    driving_durations: np.ndarray,
    order_place_durations: np.ndarray,
    actual_durations: np.ndarray
) -> Tuple[
    Annotated[RandomForestRegressor, "final_model"],
    Annotated[float, "rmse"],
    Annotated[float, "mae"],
    Annotated[float, "r2"],
]:
    """
    Train a final regression model to predict actual_total_delivery_duration.
    """
    X = np.vstack([
        prep_predictions,
        driving_durations,
        order_place_durations
    ]).T
    y = actual_durations

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("Final Regression Model Performance:")
    logger.info(f"  RMSE: {rmse}")
    logger.info(f"  MAE : {mae}")
    logger.info(f"  R²  : {r2}")

    return model, rmse, mae, r2