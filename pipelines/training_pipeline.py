from zenml import pipeline
from steps.data_loader import load_data
from steps.data_cleaner import data_cleaning_step
from steps.feature_engineering import feature_engineering_step
from steps.split_data import split_data_step
from steps.train_prep_model import train_prep_model_step
from steps.merge_predictions import merge_predictions_step
from steps.final_regression_model import final_regression_model_step
from steps.evaluate_model import evaluate_model

@pipeline(enable_cache=False)
def delivery_duration_pipeline():
    df = load_data()
    cleaned_df = data_cleaning_step(df=df)
    engineered_df = feature_engineering_step(df=cleaned_df)
    X_train, X_test, y_train, y_test = split_data_step(df=engineered_df)
    prep_time_pred, driving_durations, order_place_durations, actual_total_durations = train_prep_model_step(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test)

    final_delivery_predictions = merge_predictions_step(
    prep_duration_predictions=prep_time_pred,
    driving_durations=driving_durations,
    order_place_durations=order_place_durations,)

    final_model, rmse, mae, r2 = final_regression_model_step(
    prep_predictions=prep_time_pred,
    driving_durations=driving_durations,
    order_place_durations=order_place_durations,
    actual_durations=actual_total_durations,)

    metrics, report = evaluate_model(
    y_true=actual_total_durations,
    y_pred=final_delivery_predictions,)