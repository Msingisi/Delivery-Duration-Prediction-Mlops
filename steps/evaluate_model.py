import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Annotated, Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from zenml import step
from zenml.types import HTMLString


def generate_evaluation_html(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float]
) -> str:
    """Generates an HTML report with card-style metrics and a scatter plot."""

    # Scatter plot: Actual vs Predicted
    scatter_fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Actual Duration", "y": "Predicted Duration"},
        title="Actual vs Predicted Delivery Duration",
        opacity=0.6
    )
    scatter_fig.add_trace(
        go.Scatter(x=y_true, y=y_true, mode='lines', name='Perfect Prediction', line=dict(color='green'))
    )
    scatter_html = scatter_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # HTML report with visual cards for metrics
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                padding: 30px;
            }}
            h1 {{
                text-align: center;
                color: #333;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .card {{
                background-color: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
                font-size: 18px;
            }}
            .card h2 {{
                font-size: 24px;
                color: #2c3e50;
            }}
            .plot-section {{
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <h1>Delivery Duration Model Evaluation</h1>

        <div class="metrics-grid">
            <div class="card">
                <h2>RMSE</h2>
                <p>{metrics['rmse']:.2f}</p>
            </div>
            <div class="card">
                <h2>MAE</h2>
                <p>{metrics['mae']:.2f}</p>
            </div>
            <div class="card">
                <h2>R² Score</h2>
                <p>{metrics['r2']:.4f}</p>
            </div>
        </div>

        <div class="plot-section">
            <h2>Actual vs Predicted Plot</h2>
            {scatter_html}
        </div>
    </body>
    </html>
    """
    return html


@step
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[
    Dict[str, float],
    HTMLString
]:
    """
    Evaluate model performance and generate a report.

    Args:
        y_true: Ground truth delivery durations.
        y_pred: Predicted delivery durations.

    Returns:
        Dictionary of metrics and HTML report.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    html_report = generate_evaluation_html(y_true, y_pred, metrics)

    return metrics, HTMLString(html_report)