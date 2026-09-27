import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

DATA_DIR = "pharma_data_warehouse"

print("1. Loading materialized feature store and event logs...")
df_features = pd.read_csv(f"{DATA_DIR}/hcp_feature_store.csv")
df_claims = pd.read_csv(f"{DATA_DIR}/fct_claims.csv")
df_claims["claim_date"] = pd.to_datetime(df_claims["claim_date"])

# -----------------------------------------------------------------
# 2. Build Synthetic Causal Labels (Uplift Target Formulation)
# -----------------------------------------------------------------
# Goal: Did an HCP receive high-touch engagement (Treatment=1) vs low-touch (Control=0),
# and what was their incremental 30-day TRx lift?
print("2. Generating treatment assignments and incremental lift targets...")

# Define treatment: HCP received >= 2 rep visits in the past month
df_features["treatment_arm"] = (df_features["rep_visits_30d"] >= 2).astype(int)

# Target: 30-day forward script volume minus baseline expectation
# We simulate causal effect: Treatment adds lift predominantly to high-opportunity prescribers
np.random.seed(42)
baseline_growth = df_features["total_trx_90d"] * 0.1
causal_lift = (
    df_features["treatment_arm"] * 
    (df_features["market_decile"] * 0.4 + (1.0 - df_features["brand_market_share"]) * 3.0)
)
# Add noise and compute observed post-period brand scripts
noise = np.random.normal(0, 0.8, size=len(df_features))
df_features["observed_brand_trx_lift"] = np.maximum(0, baseline_growth + causal_lift + noise)

# -----------------------------------------------------------------
# 3. Train Two-Model Uplift Learner (T-Learner via LightGBM)
# -----------------------------------------------------------------
print("3. Training T-Learner (LightGBM Treatment & Control regressors)...")

feature_cols = [
    "market_decile", "total_trx_90d", "brand_trx_90d", "competitor_trx_90d",
    "brand_market_share", "pa_rejection_count", "claim_rejection_rate",
    "avg_copay_amount", "tier_3_script_ratio", "pa_restricted_plan_ratio",
    "total_touchpoints_30d", "emails_received_30d", "email_clicks_30d", "objection_count_30d"
]

X = df_features[feature_cols]
y = df_features["observed_brand_trx_lift"]
w = df_features["treatment_arm"]

# Split into Control (w=0) and Treatment (w=1) cohorts
X_ctrl, y_ctrl = X[w == 0], y[w == 0]
X_treat, y_treat = X[w == 1], y[w == 1]

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "verbosity": -1,
    "random_state": 42
}

model_ctrl = lgb.LGBMRegressor(**params)
model_treat = lgb.LGBMRegressor(**params)

model_ctrl.fit(X_ctrl, y_ctrl)
model_treat.fit(X_treat, y_treat)

# -----------------------------------------------------------------
# 4. Production Decision Policy Engine (Constrained Optimization)
# -----------------------------------------------------------------
class OmnichannelPolicyEngine:
    def __init__(self, m_ctrl, m_treat, feature_names):
        self.m_ctrl = m_ctrl
        self.m_treat = m_treat
        self.feature_names = feature_names

    def decide_nba(self, hcp_row: pd.Series) -> dict:
        hcp_id = hcp_row["hcp_id"]
        x_vec = pd.DataFrame([hcp_row[self.feature_names]])

        # 1. Compute predicted incremental causal lift (Uplift)
        pred_ctrl = self.m_ctrl.predict(x_vec)[0]
        pred_treat = self.m_treat.predict(x_vec)[0]
        uplift_score = float(pred_treat - pred_ctrl)

        # 2. Candidate Action Evaluation under Hard Constraints
        # Candidate set: (Channel, Content_Type, Expected_Effort_Cost)
        candidate_actions = [
            {"channel": "Rep_F2F", "content": "Efficacy_OS_PFS", "action_cost": 4.0},
            {"channel": "Rep_F2F", "content": "Copay_Access_Support", "action_cost": 3.0},
            {"channel": "Approved_Email", "content": "Efficacy_OS_PFS", "action_cost": 0.5},
            {"channel": "Approved_Email", "content": "Copay_Access_Support", "action_cost": 0.5},
            {"channel": "Webinar", "content": "Dosing_Administration", "action_cost": 1.5},
            {"channel": "No_Action", "content": "Holdout_Fatigued", "action_cost": 0.0}
        ]

        scored_actions = []

        for act in candidate_actions:
            ch = act["channel"]
            cnt = act["content"]
            
            # --- Hard Policy Filters ---
            # Rule A: Email Opt-In / Fatigue Check
            if ch == "Approved_Email":
                if not hcp_row["email_opt_in"]:
                    continue  # Regulatory compliance drop
                if hcp_row["emails_received_30d"] >= 3:
                    continue  # Fatigue limit: max 3 emails / 30d

            # Rule B: Field Rep Access Limit
            if ch == "Rep_F2F":
                if hcp_row["rep_visits_30d"] >= 2:
                    continue  # Rep access cap: max 2 visits / 30d

            # Rule C: Payer Barrier Alignment
            # If high Prior Auth rejections, prioritize access content over efficacy
            payer_friction = hcp_row["pa_restricted_plan_ratio"] > 0.40 or hcp_row["claim_rejection_rate"] > 0.25
            if payer_friction and cnt == "Efficacy_OS_PFS":
                penalty = 0.50  # Downgrade efficacy if scripts can't get reimbursed
            else:
                penalty = 1.0

            # --- Ranking Utility Function ---
            # Expected Net Margin = (Uplift * Value_Per_Script * Friction_Modifier) - Action_Cost
            value_per_trx = 8.0  # Normalized economic margin per prescription
            if ch == "No_Action":
                expected_net_value = 0.0
            else:
                expected_net_value = (uplift_score * value_per_trx * penalty) - act["action_cost"]

            scored_actions.append({
                "channel": ch,
                "content_topic": cnt,
                "predicted_uplift": round(uplift_score, 3),
                "net_expected_value": round(expected_net_value, 3),
                "payer_friction_detected": payer_friction
            })

        # Rank actions descending by net expected value
        ranked = sorted(scored_actions, key=lambda x: x["net_expected_value"], reverse=True)
        top_decision = ranked[0] if ranked else {
            "channel": "No_Action", 
            "content_topic": "All_Channels_Fatigued", 
            "predicted_uplift": 0.0, 
            "net_expected_value": 0.0,
            "payer_friction_detected": False
        }

        return {
            "hcp_id": hcp_id,
            "decile": int(hcp_row["market_decile"]),
            "preferred_channel": hcp_row["preferred_channel"],
            "top_action": top_decision,
            "alternative_options_count": len(ranked)
        }

# -----------------------------------------------------------------
# 5. Run Decision Engine on Prescriber Cohort
# -----------------------------------------------------------------
print("4. Executing NBA Policy Engine across active HCP profiles...\n")
policy_engine = OmnichannelPolicyEngine(model_ctrl, model_treat, feature_cols)

sample_profiles = df_features.sample(5, random_state=12)
for _, row in sample_profiles.iterrows():
    decision = policy_engine.decide_nba(row)
    print(f"[{decision['hcp_id']}] Decile: {decision['decile']} | Preferred: {decision['preferred_channel']}")
    print(f" -> RECOMMENDED ACTION : {decision['top_action']['channel']} ({decision['top_action']['content_topic']})")
    print(f" -> NET EXPECTED VALUE : {decision['top_action']['net_expected_value']} (Uplift: {decision['top_action']['predicted_uplift']})")
    print(f" -> PAYER BARRIER FLAG : {decision['top_action']['payer_friction_detected']}")
    print("-" * 75)

import joblib

# Save models and metadata to disk
model_artifacts = {
    "model_ctrl": model_ctrl,
    "model_treat": model_treat,
    "feature_cols": feature_cols
}
joblib.dump(model_artifacts, f"{DATA_DIR}/nba_uplift_models.joblib")
print("Models serialized to pharma_data_warehouse/nba_uplift_models.joblib")