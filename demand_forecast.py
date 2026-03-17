import pandas as pd
import numpy as np

# Optional Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing


# -----------------------------
# MODEL SELECTION (FIXED CASE)
# -----------------------------
def select_model(data):
    df = pd.DataFrame(data)  # 🔥 FIX
    n = len(df)

    if n < 30:
        return "MOVING_AVERAGE_7"
    elif n < 90:
        return "MOVING_AVERAGE_30"
    elif n >= 180 and PROPHET_AVAILABLE and df["qty_sold"].std() > 5:
        return "PROPHET"
    else:
        return "HOLT_WINTERS"


# -----------------------------
# MOVING AVERAGE (FIXED)
# -----------------------------
def fit_moving_average(data, window=7):
    df = pd.DataFrame(data)  # 🔥 FIX

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    rolling = df["qty_sold"].rolling(window).mean()

    if pd.isna(rolling.iloc[-1]):
        return df["qty_sold"].mean()

    return rolling.iloc[-1]


# -----------------------------
# HOLT-WINTERS (FIXED)
# -----------------------------
def fit_holt_winters(data):
    df = pd.DataFrame(data)  # 🔥 FIX

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    series = df["qty_sold"]

    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None
        )
        fit = model.fit()

        forecast = fit.forecast(30)

        return forecast.mean()

    except Exception:
        return series.mean()


# -----------------------------
# PROPHET (FIXED)
# -----------------------------
def fit_prophet(data):
    df = pd.DataFrame(data)  # 🔥 FIX

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if not PROPHET_AVAILABLE:
        return df["qty_sold"].mean()

    prophet_df = df.rename(columns={"date": "ds", "qty_sold": "y"})

    try:
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        return forecast["yhat"].tail(30).mean()

    except Exception:
        return df["qty_sold"].mean()


# -----------------------------
# MAPE
# -----------------------------
def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    actual = np.where(actual == 0, 1, actual)

    return np.mean(np.abs((actual - predicted) / actual)) * 100


# -----------------------------
# REORDER
# -----------------------------
def calculate_reorder(avg_demand, stock_on_hand, lead_time_days=5):
    safety_stock = avg_demand * lead_time_days * 1.25
    reorder_point = avg_demand * lead_time_days + safety_stock

    days_left = stock_on_hand / avg_demand if avg_demand > 0 else 0
    reorder_needed = stock_on_hand < reorder_point

    return {
        "avg_daily_demand": round(avg_demand, 2),
        "days_left": round(days_left, 2),
        "reorder_point": round(reorder_point, 2),
        "reorder_needed": reorder_needed,
        "suggested_qty": int(reorder_point - stock_on_hand) if reorder_needed else 0
    }