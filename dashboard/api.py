"""
FastAPI backend for Battery SOH GPR inference.
Run: uvicorn dashboard.api:app --reload
"""

import os
import sys
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

app = FastAPI(
    title="Battery SOH GPR API",
    description="Gaussian Process Regression inference for battery State of Health estimation.",
    version="1.0.0",
)

SUPPORTED_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    battery_id: str = Field(..., example="B0005", description="Battery identifier")
    cycles: List[int] = Field(..., example=[50, 100, 150], description="Cycle numbers to predict SOH for")


class PredictionPoint(BaseModel):
    cycle: int
    soh_pred: float
    soh_std: float
    ci_lower: float
    ci_upper: float


class PredictResponse(BaseModel):
    battery_id: str
    predictions: List[PredictionPoint]


class BatteryInfo(BaseModel):
    battery_id: str
    model_available: bool
    supported: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "Battery SOH GPR API"}


@app.get("/batteries", response_model=List[BatteryInfo], summary="List available batteries")
def list_batteries():
    from gpr_model import load_model
    result = []
    for bid in SUPPORTED_BATTERIES:
        try:
            load_model(bid)
            available = True
        except FileNotFoundError:
            available = False
        result.append(BatteryInfo(battery_id=bid, model_available=available, supported=True))
    return result


@app.post("/predict", response_model=PredictResponse, summary="Predict SOH for given cycles")
def predict(req: PredictRequest):
    if req.battery_id not in SUPPORTED_BATTERIES:
        raise HTTPException(status_code=400, detail=f"Unsupported battery_id. Choose from {SUPPORTED_BATTERIES}")
    if not req.cycles:
        raise HTTPException(status_code=400, detail="cycles list must not be empty")
    if any(c < 1 for c in req.cycles):
        raise HTTPException(status_code=400, detail="All cycle numbers must be >= 1")

    try:
        from gpr_model import predict_soh
        soh_preds, soh_stds = predict_soh(req.battery_id, req.cycles)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No trained model found for {req.battery_id}. Run training first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    predictions = [
        PredictionPoint(
            cycle=c,
            soh_pred=round(p, 6),
            soh_std=round(s, 6),
            ci_lower=round(p - 2 * s, 6),
            ci_upper=round(p + 2 * s, 6),
        )
        for c, p, s in zip(req.cycles, soh_preds, soh_stds)
    ]
    return PredictResponse(battery_id=req.battery_id, predictions=predictions)


@app.get("/predict/{battery_id}/{cycle}", response_model=PredictionPoint, summary="Predict SOH for a single cycle")
def predict_single(battery_id: str, cycle: int):
    resp = predict(PredictRequest(battery_id=battery_id, cycles=[cycle]))
    return resp.predictions[0]


@app.get("/results/{battery_id}", summary="Get full prediction results CSV as JSON")
def get_results(battery_id: str):
    import pandas as pd
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    path = os.path.join(results_dir, f"{battery_id}_predictions.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No results found for {battery_id}")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
