from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import pandas as pd
import io

from run_pipeline import process_sku

app = FastAPI()

# 🔐 API KEY
API_KEY = "supply_2026"


@app.get("/")
def home():
    return {"message": "Demand Forecast API running"}


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    # 🔒 API KEY VALIDATION
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Read CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Validate required columns
    required_cols = ["date", "sku_code", "qty_sold", "stock_on_hand", "lead_time_days"]
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    results = []

    for sku, group in df.groupby("sku_code"):
        res = process_sku(sku, group)
        results.append(res)

    return results


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=10000)