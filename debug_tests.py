# -*- coding: utf-8 -*-
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("sutram.demand_forecast").setLevel(logging.WARNING)

from datetime import date, timedelta
import numpy as np
from demand_forecast import CONFIG, detect_seasonality, select_model, fit_moving_average, calculate_reorder

def make_history(n, base_qty=10, pattern=None, start_date=date(2024, 1, 1)):
    rows = []
    for i in range(n):
        d = start_date + timedelta(days=i)
        qty = pattern[i % len(pattern)] if pattern else base_qty
        rows.append({"date": d.strftime("%Y-%m-%d"), "qty_sold": qty})
    return rows

# Run each test individually and print results
tests = []

# Test 1
r1 = select_model(make_history(20))
tests.append(("T1: 20 rows -> MA_7", r1 == "MOVING_AVERAGE_7", r1))

# Test 2
r2 = select_model(make_history(60, base_qty=50))
tests.append(("T2: 60 rows flat -> MA_30", r2 == "MOVING_AVERAGE_30", r2))

# Test 3
np.random.seed(42)
h3 = [{"date": (date(2024,1,1)+timedelta(days=i)).strftime("%Y-%m-%d"),
       "qty_sold": int(50+np.random.normal(0,2))} for i in range(100)]
r3 = select_model(h3)
q3 = [max(0, r["qty_sold"]) for r in h3]
s3 = detect_seasonality(q3)
tests.append(("T3: 100 no season -> HW", r3 == "HOLT_WINTERS", f"{r3} seasonal={s3}"))

# Test 4
weekly = [10,20,30,40,50,60,70]
h4 = make_history(200, pattern=weekly)
r4 = select_model(h4)
q4 = [max(0, r["qty_sold"]) for r in h4]
s4 = detect_seasonality(q4)
tests.append(("T4: 200 rows 7d -> PROPHET", r4 == "PROPHET", f"{r4} seasonal={s4}"))

# Test 5
try:
    ma = fit_moving_average([], window=7)
    tests.append(("T5: empty -> 30 rows", len(ma)==30, str(len(ma))))
except Exception as e:
    tests.append(("T5: empty -> 30 rows", False, str(e)))

# Test 6
r6 = calculate_reorder(make_history(60, base_qty=0), stock_on_hand=100, lead_time_days=14)
tests.append(("T6: zeros -> dl=999", r6["days_left"]==999, str(r6["days_left"])))

# Test 7
r7 = calculate_reorder(make_history(30, base_qty=40), stock_on_hand=120, lead_time_days=5)
ok7 = (r7["reorder_point"]==250.0 and r7["days_left"]==3 and
       r7["suggested_qty"]==1280 and r7["reorder_needed"]==True and r7["safety_stock"]==50.0)
tests.append(("T7: BRK-4521", ok7, str(r7)))

# Test 8
try:
    h8 = [{"date": (date(2024,1,1)+timedelta(days=i)).strftime("%Y-%m-%d"),
           "qty_sold": -10} for i in range(100)]
    r8m = select_model(h8)
    r8r = calculate_reorder(h8, stock_on_hand=50, lead_time_days=7)
    ok8 = r8r["days_left"]==999 and r8r["reorder_needed"]==False
    tests.append(("T8: negative qty", ok8, f"dl={r8r['days_left']} rn={r8r['reorder_needed']}"))
except Exception as e:
    tests.append(("T8: negative qty", False, str(e)))

print("="*60)
for name, ok, detail in tests:
    print(f"  {'PASS' if ok else 'FAIL'} | {name} | {detail}")
p = sum(1 for _,ok,_ in tests if ok)
print(f"\n{p}/{len(tests)} passed")
