import logging
import numpy as np
from typing import Annotated
from zenml import step

logger = logging.getLogger(__name__)

@step
def merge_predictions_step(
    prep_duration_predictions: np.ndarray,
    driving_durations: np.ndarray,
    order_place_durations: np.ndarray,
) -> Annotated[np.ndarray, "final_delivery_predictions"]:
    """
    Merge predicted prep time with estimated driving and order placement durations
    to form the final predicted total delivery duration.

    Returns:
        final_delivery_predictions: Predicted total delivery durations (in seconds)
    """
    final_predictions = (
        prep_duration_predictions +
        driving_durations +
        order_place_durations
    )

    logger.info(f"Average predicted delivery duration: {np.mean(final_predictions):.2f} seconds")
    logger.info(f"Min/Max: {np.min(final_predictions)} / {np.max(final_predictions)}")
    logger.info(f"Final predictions shape: {final_predictions.shape}")
    
    return final_predictions