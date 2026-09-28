"""Bank Customer Churn API — a thin FastAPI wrapper around the trained model.

Deliberately simple: one /predict endpoint. Callers send ordinary customer
attributes; the engineered features are rebuilt server-side so the request
matches the exact feature set the model was trained on.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib

bundle = joblib.load("churn_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
MODEL_NAME = bundle["model_name"]

app = FastAPI(title="Bank Customer Churn API", version="1.0.0",
              description="Predicts the probability a bank customer churns, with drivers.")


class Customer(BaseModel):
    credit_score: int = Field(..., ge=350, le=850, examples=[650])
    geography: Literal["France", "Germany", "Spain"] = Field(..., examples=["Germany"])
    gender: Literal["Male", "Female"] = Field(..., examples=["Female"])
    age: int = Field(..., ge=18, le=100, examples=[45])
    tenure: int = Field(..., ge=0, le=10, examples=[3])
    balance: float = Field(..., ge=0, examples=[120000.0])
    num_of_products: int = Field(..., ge=1, le=4, examples=[1])
    has_cr_card: int = Field(..., ge=0, le=1, examples=[1])
    is_active_member: int = Field(..., ge=0, le=1, examples=[0])
    estimated_salary: float = Field(..., gt=0, examples=[95000.0])


class Prediction(BaseModel):
    churn_probability: float
    risk_level: str
    will_churn: bool
    top_factors: list[str]
    model_name: str


def build_feature_row(c: Customer) -> list[float]:
    """Recreate the exact engineered feature set (and order) used in training."""
    values = {
        "CreditScore": c.credit_score,
        "Age": c.age,
        "Tenure": c.tenure,
        "Balance": c.balance,
        "NumOfProducts": c.num_of_products,
        "HasCrCard": c.has_cr_card,
        "IsActiveMember": c.is_active_member,
        "EstimatedSalary": c.estimated_salary,
        "ZeroBalance": 1 if c.balance == 0 else 0,
        "BalanceToSalary": round(c.balance / c.estimated_salary, 4),
        "ProductsPerTenure": round(c.num_of_products / (c.tenure + 1), 4),
        "EngagementScore": c.is_active_member + c.has_cr_card,
        "Geography_Germany": 1 if c.geography == "Germany" else 0,
        "Geography_Spain": 1 if c.geography == "Spain" else 0,
        "Gender_Male": 1 if c.gender == "Male" else 0,
    }
    return [float(values[f]) for f in FEATURES]  # FEATURES preserves training order


def explain(c: Customer) -> list[str]:
    """Plain-language drivers from domain rules (model-agnostic, matches EDA)."""
    notes = []
    if c.is_active_member == 0:
        notes.append("Inactive member — the strongest actionable risk factor")
    if c.geography == "Germany":
        notes.append("Based in Germany, which has the highest churn rate")
    if c.age >= 50:
        notes.append(f"Age {c.age} — churn risk rises sharply after 50")
    if c.num_of_products >= 3:
        notes.append(f"Holds {c.num_of_products} products — this group churns heavily")
    if c.tenure <= 1:
        notes.append("Very new customer — limited relationship built yet")
    if c.is_active_member == 1:
        notes.append("Active member — the strongest protective factor")
    return notes or ["No single dominant factor; risk comes from the overall profile."]


@app.get("/")
def home():
    return {"status": "ok", "model": MODEL_NAME, "n_features": len(FEATURES), "docs": "/docs"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    row = build_feature_row(customer)
    proba = float(model.predict_proba([row])[0, 1])
    level = "high" if proba >= 0.5 else "medium" if proba >= 0.25 else "low"
    return Prediction(
        churn_probability=round(proba, 4),
        risk_level=level,
        will_churn=proba >= 0.5,
        top_factors=explain(customer),
        model_name=MODEL_NAME,
    )
