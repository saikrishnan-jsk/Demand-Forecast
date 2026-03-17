"""
Generate a realistic sample inventory.csv for the demand forecasting pipeline.
Creates 6 SKUs with varied history lengths to exercise all model paths.
"""

import csv
import math
import random
from datetime import date, timedelta

random.seed(42)

END_DATE = date(2026, 3, 17)

SKUS = [
    # (sku_code, days_of_data, base_qty, pattern_type, stock_on_hand, lead_time_days)
    ("BRK-4521",  15,  40, "flat",       120,  5),   # < 30 → MA-7
    ("CLT-1102",  60,  25, "trend_up",   300, 10),   # < 90 → MA-30
    ("ENG-3300", 100,  50, "noisy",      800, 14),   # ≥ 90 → Holt-Winters
    ("FLT-2200", 120,  30, "mild_trend", 150,  7),   # ≥ 90 → Holt-Winters
    ("GKT-5500", 210,  45, "seasonal",   500, 10),   # ≥ 90 → Holt-Winters/Prophet
    (" SPK-7700 ", 50, 20, "flat",        60,  7),   # spaces in sku_code, < 90 → MA-30
]

rows = []

for sku_code, n_days, base_qty, pattern, stock, lead_time in SKUS:
    start = END_DATE - timedelta(days=n_days - 1)
    for i in range(n_days):
        d = start + timedelta(days=i)
        if pattern == "flat":
            qty = base_qty + random.randint(-3, 3)
        elif pattern == "trend_up":
            qty = base_qty + int(i * 0.3) + random.randint(-5, 5)
        elif pattern == "noisy":
            qty = base_qty + random.randint(-15, 15)
        elif pattern == "mild_trend":
            qty = base_qty + int(i * 0.15) + random.randint(-4, 4)
        elif pattern == "seasonal":
            qty = base_qty + int(20 * math.sin(2 * math.pi * i / 7)) + random.randint(-3, 3)
        else:
            qty = base_qty

        qty = max(0, qty)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "sku_code": sku_code,
            "qty_sold": qty,
            "stock_on_hand": stock,
            "lead_time_days": lead_time,
        })

with open("d:/Demand prediction AI/inventory.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "sku_code", "qty_sold", "stock_on_hand", "lead_time_days"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated inventory.csv with {len(rows)} rows across {len(SKUS)} SKUs")
for sku_code, n_days, *_ in SKUS:
    print(f"  {sku_code.strip():>10}: {n_days} days")
