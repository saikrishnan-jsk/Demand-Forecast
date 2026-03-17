from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

from run_pipeline import process_sku

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Demand Forecast API running"}


@app.post("/forecast")
async def forecast(file: UploadFile = File(...)):

    # Read CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    results = []

    for sku, group in df.groupby("sku_code"):
        res = process_sku(sku, group)
        results.append(res)

    result_df = pd.DataFrame(results)

    return result_df.to_dict(orient="records")


# -----------------------------
# RUN SERVER (NO INDENTATION)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=10000)