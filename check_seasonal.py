import sys
import logging
logging.disable(logging.CRITICAL)

from datetime import date, timedelta
import numpy as np
from demand_forecast import CONFIG, detect_seasonality, select_model

# Test 4 - sinusoidal but length is exact multiple of 7 (210 points)
h4 = [{"date": (date(2024,1,1)+timedelta(days=i)).strftime("%Y-%m-%d"), "qty_sold": int(50+40*np.sin(2*np.pi*i/7))} for i in range(210)]

qty4 = [max(0, r["qty_sold"]) for r in h4]
seasonal = detect_seasonality(qty4)
model = select_model(h4)

series = np.array(qty4, dtype=float)
detrended = series - series.mean()
fft_vals = np.fft.rfft(detrended)
fft_vals[0] = 0.0
mags = np.abs(fft_vals)
total = float(mags.sum())
dom = float(mags.max())
strength = dom/total if total>0 else 0

print(f"seasonal={seasonal}")
print(f"model={model}")
print(f"strength={strength}")
print(f"n={len(h4)}")
print(f"pass={model=='PROPHET'}")
