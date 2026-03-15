from fastapi import FastAPI
import pandas as pd

app = FastAPI()

suppliers = pd.read_csv("data/suppliers.csv")
esg = pd.read_csv("data/esg_metrics.csv")
transactions = pd.read_csv("data/transactions.csv")


@app.get("/")
def home():
    return {"message": "Supplier AI Intelligence API running"}


@app.get("/suppliers")
def get_suppliers():
    return suppliers.to_dict(orient="records")


@app.get("/esg")
def get_esg():
    return esg.to_dict(orient="records")


@app.get("/transactions")
def get_transactions():
    return transactions.head(100).to_dict(orient="records")


@app.get("/supplier_performance")
def supplier_performance():

    perf = transactions.groupby("supplier_id").agg(
        avg_delay=("delivery_delay_days","mean"),
        avg_defect=("defect_rate","mean"),
        avg_cost_variance=("cost_variance","mean")
    ).reset_index()

    perf["risk_score"] = (
        perf["avg_delay"] * 0.4 +
        perf["avg_defect"] * 100 * 0.4 +
        abs(perf["avg_cost_variance"]) * 0.2
    )

    return perf.to_dict(orient="records")