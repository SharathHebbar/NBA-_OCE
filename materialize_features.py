import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "pharma_data_warehouse"

print("Loading raw CSV files...")
df_hcp = pd.read_csv(f"{DATA_DIR}/dim_hcp.csv")
df_payer = pd.read_csv(f"{DATA_DIR}/dim_payer.csv")
df_patient = pd.read_csv(f"{DATA_DIR}/dim_patient.csv")
df_claims = pd.read_csv(f"{DATA_DIR}/fct_claims.csv")
df_engagement = pd.read_csv(f"{DATA_DIR}/fct_engagement.csv")

df_claims["claim_date"] = pd.to_datetime(df_claims["claim_date"])
df_engagement["timestamp"] = pd.to_datetime(df_engagement["timestamp"])
now = df_claims["claim_date"].max() # reference execution anchor

# -----------------------------------------------------------------
# 1. Prescribing & Market Share Metrics (Last 90 Days)
# -----------------------------------------------------------------
print("Aggregating claims & market metrics...")
cutoff_90d = now - timedelta(days=90)
claims_90d = df_claims[df_claims["claim_date"] >= cutoff_90d]

claims_features = claims_90d.groupby("prescribing_hcp_id").agg(
    total_trx_90d=("claim_id", "count"),
    brand_trx_90d=("ndc_product", lambda x: (x == "Our_Brand").sum()),
    competitor_trx_90d=("ndc_product", lambda x: (x != "Our_Brand").sum()),
    pa_rejection_count=("rejection_reason", lambda x: (x == "PRIOR_AUTH_DENIAL").sum()),
    claim_rejection_rate=("claim_status", lambda x: (x == "REJECTED").mean())
).reset_index().rename(columns={"prescribing_hcp_id": "hcp_id"})

claims_features["brand_market_share"] = (
    claims_features["brand_trx_90d"] / claims_features["total_trx_90d"].replace(0, np.nan)
).fillna(0.0)

# -----------------------------------------------------------------
# 2. Payer Landscape & Patient Burden per Prescriber
# -----------------------------------------------------------------
print("Computing payer barrier indicators...")
claims_payer = df_claims.merge(df_payer, on="payer_id", how="left")
payer_features = claims_payer.groupby("prescribing_hcp_id").agg(
    avg_copay_amount=("patient_copay_amount", "mean"),
    tier_3_script_ratio=("formulary_tier", lambda x: (x == 3).mean()),
    pa_restricted_plan_ratio=("pa_required", "mean")
).reset_index().rename(columns={"prescribing_hcp_id": "hcp_id"})

# -----------------------------------------------------------------
# 3. Omnichannel Engagement & Fatigue Windows (Last 30 Days)
# -----------------------------------------------------------------
print("Computing channel activity & fatigue...")
cutoff_30d = now - timedelta(days=30)
eng_30d = df_engagement[df_engagement["timestamp"] >= cutoff_30d]

eng_features = eng_30d.groupby("hcp_id").agg(
    total_touchpoints_30d=("engagement_id", "count"),
    rep_visits_30d=("channel", lambda x: x.isin(["Rep_F2F", "Rep_Virtual"]).sum()),
    emails_received_30d=("channel", lambda x: (x == "Approved_Email").sum()),
    email_clicks_30d=("interaction_type", lambda x: (x == "CLICKED").sum()),
    samples_received_30d=("samples_dropped_units", "sum"),
    objection_count_30d=("sentiment_feedback", lambda x: (x == "Access_Objection").sum())
).reset_index()

# -----------------------------------------------------------------
# 4. Master Feature Vector Synthesis
# -----------------------------------------------------------------
print("Joining into master feature vector...")
feature_store = df_hcp.merge(claims_features, on="hcp_id", how="left")
feature_store = feature_store.merge(payer_features, on="hcp_id", how="left")
feature_store = feature_store.merge(eng_features, on="hcp_id", how="left")

# Impute zero-activity entities
fill_zeros = [
    "total_trx_90d", "brand_trx_90d", "competitor_trx_90d", "brand_market_share",
    "pa_rejection_count", "claim_rejection_rate", "avg_copay_amount",
    "tier_3_script_ratio", "pa_restricted_plan_ratio", "total_touchpoints_30d",
    "rep_visits_30d", "emails_received_30d", "email_clicks_30d", "samples_received_30d",
    "objection_count_30d"
]
feature_store[fill_zeros] = feature_store[fill_zeros].fillna(0.0)

# Save offline training store & online ready format
# feature_store.to_parquet(f"{DATA_DIR}/hcp_feature_store.parquet", index=False)
feature_store.to_csv(f"{DATA_DIR}/hcp_feature_store.csv", index=False)

print("\nFeature Store Materialized Successfully.")
print(f"Total HCP Entities: {len(feature_store)}")
print(f"Total Feature Dimensions: {feature_store.shape[1]}")
print(feature_store[["hcp_id", "market_decile", "brand_market_share", "claim_rejection_rate", "rep_visits_30d"]].head())