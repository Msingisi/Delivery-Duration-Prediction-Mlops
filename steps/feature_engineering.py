import logging
import pandas as pd
import numpy as np
from zenml import step
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

@step
def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering: ratios, dummies, drop collinears, and reduce VIF.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Feature-engineered DataFrame.
    """
    # -- FEATURE ENGINEERING --

    df["busy_dashers_ratio"] = df["total_busy_dashers"] / df["total_onshift_dashers"]
    df["price_range_of_items"] = df["max_item_price"] - df["min_item_price"]
    df["avg_price_per_item"] = df["subtotal"] / df["total_items"]
    df["percent_distinct_item_of_total"] = df["num_distinct_items"] / df["total_items"]
    df["estimated_non_prep_duration"] = df["estimated_store_to_consumer_driving_duration"] + df["estimated_order_place_duration"]

    # -- HANDLE CATEGORICALS --

    store_id_mode_map = (
        df.groupby("store_id")["store_primary_category"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .to_dict()
    )
    df["store_primary_category"] = df.apply(
        lambda row: store_id_mode_map.get(row["store_id"], np.nan)
        if pd.isna(row["store_primary_category"])
        else row["store_primary_category"],
        axis=1,
    )

    # Dummies
    order_protocol_dummies = pd.get_dummies(df["order_protocol"], prefix="order_protocol")
    store_category_dummies = pd.get_dummies(df["store_primary_category"], prefix="category")

    # Concat dummies
    df = pd.concat([df, order_protocol_dummies, store_category_dummies], axis=1)

    # Drop original categorical columns and identifiers
    df.drop(
        columns=[
            "created_at",
            "actual_delivery_time",
            "order_protocol",
            "store_primary_category",
            "store_id",
            "market_id",
            "nan_free_store_primary_category",
        ],
        inplace=True,
        errors="ignore",
    )

    # Drop redundant features
    collinear_to_drop = [
        "total_onshift_dashers",
        "total_busy_dashers",
        "category_indonesian",
        "estimated_non_prep_duration",
        "subtotal",
        "num_distinct_items",
        "max_item_price",
        "min_item_price",
    ]
    df.drop(columns=[col for col in collinear_to_drop if col in df.columns], inplace=True)

    # -- VIF FILTERING --

    def compute_vif(df_: pd.DataFrame) -> pd.DataFrame:
        features = df_.columns
        vif_data = pd.DataFrame()
        vif_data["feature"] = features
        vif_data["VIF"] = [
            variance_inflation_factor(df_[features].values, i) for i in range(len(features))
        ]
        return vif_data.sort_values(by="VIF").reset_index(drop=True)

    features = df.drop(columns=["actual_total_delivery_duration"], errors="ignore")
    features = features.select_dtypes(include=[np.number])
    features.fillna(0, inplace=True)

    dropped_features = []
    while True:
        vif_result = compute_vif(features)
        if vif_result["VIF"].iloc[-1] > 20:
            to_drop = vif_result["feature"].iloc[-1]
            features.drop(columns=[to_drop], inplace=True)
            dropped_features.append(to_drop)
            logger.info(f"Dropping high VIF column: {to_drop}")
        else:
            break

    if dropped_features:
        logger.info(f"Total features dropped due to high VIF: {len(dropped_features)}")
    else:
        logger.info("No features dropped due to VIF.")

    selected_features = features.columns.tolist()
    final_df = df[selected_features + ["actual_total_delivery_duration"]]

    # Replace inf and drop remaining nulls
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True)

    logger.info(f"Selected features after VIF filtering: {len(selected_features)}")
    logger.info(f"Final shape of engineered DataFrame: {final_df.shape}")

    return final_df