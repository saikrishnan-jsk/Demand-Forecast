"""
SUTRAM — Demand Forecasting Pipeline (FINAL CLEAN)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast import (
    fit_holt_winters,
    fit_moving_average,
    fit_prophet,
    select_model,
)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("pipeline")

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = Path("inventory.csv")
OUTPUT_PATH = Path("forecast_results.csv")


# -----------------------------
# Load + Clean Data
# -----------------------------
def load_data():
    df = pd.read_csv(CSV_PATH)

    df["sku_code"] = df["sku_code"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"])

    df["qty_sold"] = pd.to_numeric(df["qty_sold"], errors="coerce").fillna(0)
    df["stock_on_hand"] = pd.to_numeric(df["stock_on_hand"], errors="coerce").fillna(0)
    df["lead_time_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce").fillna(7)

    return df


# -----------------------------
# Forecast per SKU
# -----------------------------
def process_sku(sku, group):

    forecast = None
    model = None

    group = group.sort_values("date")

    history = [
        {"date": row["date"].strftime("%Y-%m-%d"), "qty_sold": int(row["qty_sold"])}
        for _, row in group.iterrows()
    ]

    # -----------------------------
    # Model selection + forecast
    # -----------------------------
    try:
        model = select_model(history)

        if model == "MOVING_AVERAGE_7":
            forecast = fit_moving_average(history, window=7)

        elif model == "MOVING_AVERAGE_30":
            forecast = fit_moving_average(history, window=30)

        elif model == "HOLT_WINTERS":
            forecast = fit_holt_winters(history)

        elif model == "PROPHET":
            forecast = fit_prophet(history)

        else:
            forecast = fit_moving_average(history, window=7)
            model = "MOVING_AVERAGE_7"

    except Exception as e:
        logger.warning(f"{sku} error → fallback MA7 | {e}")
        forecast = fit_moving_average(history, window=7)
        model = "MOVING_AVERAGE_7"

    if forecast is None:
        raise ValueError(f"Forecast NOT generated for SKU: {sku}")

    # -----------------------------
    # Extract predictions
    # -----------------------------
    if isinstance(forecast, list):
        preds = [f["predicted_demand"] for f in forecast]
    else:
        preds = [forecast]

    avg_demand = float(np.mean(preds))
    avg_demand = min(avg_demand, group["qty_sold"].max() * 1.5)

    # -----------------------------
    # Stock + lead time
    # -----------------------------
    latest = group.iloc[-1]
    stock = int(latest["stock_on_hand"])
    lead_time = int(latest["lead_time_days"])

    # -----------------------------
    # Reorder logic (fixed)
    # -----------------------------
    safety_stock = np.std(group["qty_sold"]) * np.sqrt(lead_time)
    reorder_point = (avg_demand * lead_time) + safety_stock

    if avg_demand > 0:
        days_left = math.floor(stock / avg_demand)
    else:
        days_left = 999

    reorder_needed = stock < reorder_point

    # -----------------------------
    # REAL MAPE (Train/Test Split)
    # -----------------------------
    split = int(len(group) * 0.8)

    train_df = group.iloc[:split]
    test_df = group.iloc[split:]

    if len(test_df) > 0:

        train_history = [
            {"date": row["date"].strftime("%Y-%m-%d"), "qty_sold": int(row["qty_sold"])}
            for _, row in train_df.iterrows()
        ]

        try:
            if model == "MOVING_AVERAGE_7":
                pred_value = fit_moving_average(train_history, window=7)

            elif model == "MOVING_AVERAGE_30":
                pred_value = fit_moving_average(train_history, window=30)

            elif model == "HOLT_WINTERS":
                pred_value = fit_holt_winters(train_history)

            elif model == "PROPHET":
                pred_value = fit_prophet(train_history)

            else:
                pred_value = fit_moving_average(train_history, window=7)

        except:
            pred_value = np.mean(train_df["qty_sold"])

        actual = test_df["qty_sold"].values
        predicted = np.array([pred_value] * len(actual))

        actual = np.where(actual == 0, 1, actual)

        mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    else:
        mape = None

    # -----------------------------
    # Final Output
    # -----------------------------
    return {
        "sku_code": sku,
        "model_used": model,
        "avg_demand": round(avg_demand, 2),
        "total_stock": stock,
        "lead_time_days": lead_time,
        "reorder_point": round(reorder_point, 2),
        "days_left": days_left,
        "reorder_needed": reorder_needed,
        "mape": round(mape, 2) if mape else None,
    }


# -----------------------------
# Main Pipeline
# -----------------------------
def run():

    print("\n=== DEMAND FORECAST PIPELINE ===\n")

    df = load_data()

    results = []

    for sku, group in df.groupby("sku_code"):
        res = process_sku(sku, group)
        results.append(res)

    result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_PATH, index=False)

    print(result_df.to_string(index=False))
    print("\nSaved → forecast_results.csv\n")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run()