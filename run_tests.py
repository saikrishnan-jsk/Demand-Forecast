# -*- coding: ascii -*-
import sys
import logging
logging.disable(logging.CRITICAL)

from datetime import date, timedelta
import numpy as np
from demand_forecast import CONFIG, detect_seasonality, select_model, fit_moving_average, calculate_reorder

def make_history(n, base_qty=10, start_date=date(2024, 1, 1)):
    return [{"date": (start_date + timedelta(days=i)).strftime("%Y-%m-%d"), "qty_sold": base_qty} for i in range(n)]

results = []

# T1
r = select_model(make_history(20))
results.append(f"T1: {r == 'MOVING_AVERAGE_7'} got={r}")

# T2
r = select_model(make_history(60, base_qty=50))
results.append(f"T2: {r == 'MOVING_AVERAGE_30'} got={r}")

# T3
np.random.seed(42)
h3 = [{"date": (date(2024,1,1)+timedelta(days=i)).strftime("%Y-%m-%d"), "qty_sold": int(50+np.random.normal(0,2))} for i in range(100)]
r = select_model(h3)
results.append(f"T3: {r == 'HOLT_WINTERS'} got={r}")

# T4 - sinusoidal
h4 = [{"date": (date(2024,1,1)+timedelta(days=i)).strftime("%Y-%m-%d"), "qty_sold": int(50+40*np.sin(2*np.pi*i/7))} for i in range(200)]
r = select_model(h4)
results.append(f"T4: {r == 'PROPHET'} got={r}")

# T5
ma = fit_moving_average([], window=7)
results.append(f"T5: {len(ma)==30} rows={len(ma)}")

# T6
r6 = calculate_reorder(make_history(60, base_qty=0), 100, 14)
results.append(f"T6: {r6['days_left']==999} dl={r6['days_left']}")

# T7
r7 = calculate_reorder(make_history(30, base_qty=40), 120, 5)
ok7 = r7["reorder_point"]==250.0 and r7["days_left"]==3 and r7["suggested_qty"]==1280 and r7["reorder_needed"]==True and r7["safety_stock"]==50.0
results.append(f"T7: {ok7} rp={r7['reorder_point']} dl={r7['days_left']} sq={r7['suggested_qty']} rn={r7['reorder_needed']} ss={r7['safety_stock']}")

# T8
h8 = [{"date": (date(2024,1,1)+timedelta(days=i)).strftime("%Y-%m-%d"), "qty_sold": -10} for i in range(100)]
select_model(h8)
r8 = calculate_reorder(h8, 50, 7)
ok8 = r8["days_left"]==999 and r8["reorder_needed"]==False
results.append(f"T8: {ok8} dl={r8['days_left']} rn={r8['reorder_needed']}")

for r in results:
    print(r)

p = sum(1 for r in results if "True" in r.split(" ")[1])
print(f"TOTAL: {p}/8")
sys.exit(0 if p == 8 else 1)
