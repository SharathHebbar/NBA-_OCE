from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional, List

DATA_DIR = "pharma_data_warehouse"

app = FastAPI(title="Commercial Pharma NBA Engine", version="1.0.0")

# Load artifacts into application state
artifacts = joblib.load(f"{DATA_DIR}/nba_uplift_models.joblib")
m_ctrl = artifacts["model_ctrl"]
m_treat = artifacts["model_treat"]
feature_cols = artifacts["feature_cols"]

# In-memory feature table for fast lookup (<5ms)
feature_df = pd.read_csv(f"{DATA_DIR}/hcp_feature_store.csv").set_index("hcp_id")

class DecisionRequest(BaseModel):
    hcp_id: str

class DecisionResponse(BaseModel):
    hcp_id: str
    decile: int
    preferred_channel: str
    recommended_channel: str
    recommended_content: str
    predicted_uplift: float
    net_expected_value: float
    payer_friction_detected: bool

def compute_decision(hcp_id: str) -> dict:
    if hcp_id not in feature_df.index:
        raise ValueError(f"HCP '{hcp_id}' not found in registry.")

    row = feature_df.loc[hcp_id]
    x_vec = pd.DataFrame([row[feature_cols]])

    uplift = float(m_treat.predict(x_vec)[0] - m_ctrl.predict(x_vec)[0])

    candidate_actions = [
        {"channel": "Rep_F2F", "content": "Efficacy_OS_PFS", "action_cost": 4.0},
        {"channel": "Rep_F2F", "content": "Copay_Access_Support", "action_cost": 3.0},
        {"channel": "Approved_Email", "content": "Efficacy_OS_PFS", "action_cost": 0.5},
        {"channel": "Approved_Email", "content": "Copay_Access_Support", "action_cost": 0.5},
        {"channel": "Webinar", "content": "Dosing_Administration", "action_cost": 1.5},
    ]

    scored = []
    payer_friction = bool(row["pa_restricted_plan_ratio"] > 0.40 or row["claim_rejection_rate"] > 0.25)

    for act in candidate_actions:
        ch = act["channel"]
        cnt = act["content"]

        # Hard guardrails
        if ch == "Approved_Email" and (not row["email_opt_in"] or row["emails_received_30d"] >= 3):
            continue
        if ch == "Rep_F2F" and row["rep_visits_30d"] >= 2:
            continue

        penalty = 0.50 if (payer_friction and cnt == "Efficacy_OS_PFS") else 1.0
        net_val = (uplift * 8.0 * penalty) - act["action_cost"]

        scored.append((net_val, ch, cnt))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_val, best_ch, best_cnt = scored[0]
    else:
        best_val, best_ch, best_cnt = 0.0, "No_Action", "Fatigued_All_Channels"

    return {
        "hcp_id": hcp_id,
        "decile": int(row["market_decile"]),
        "preferred_channel": str(row["preferred_channel"]),
        "recommended_channel": best_ch,
        "recommended_content": best_cnt,
        "predicted_uplift": round(uplift, 3),
        "net_expected_value": round(best_val, 3),
        "payer_friction_detected": payer_friction
    }

@app.post("/v1/nba/recommend", response_model=DecisionResponse)
def get_single_hcp_nba(req: DecisionRequest):
    try:
        return compute_decision(req.hcp_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/v1/territory/{territory_id}/plan")
def get_territory_batch(territory_id: str, limit: int = 50):
    subset = feature_df[feature_df["territory_id"] == territory_id].index[:limit]
    return [compute_decision(hid) for hid in subset]