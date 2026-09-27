import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Commercial Omnichannel NBA Portal", layout="wide")
st.title("Commercial Pharma — Field & Omnichannel NBA Engine")

# Fetch local feature store for the selector dropdown
df_features = pd.read_csv("pharma_data_warehouse/hcp_feature_store.csv")

st.sidebar.header("Navigation & Territory Filter")
selected_territory = st.sidebar.selectbox("Select Sales Territory", df_features["territory_id"].unique())

territory_hcps = df_features[df_features["territory_id"] == selected_territory]
selected_hcp_id = st.sidebar.selectbox("Select Target Prescriber", territory_hcps["hcp_id"].tolist())

# Inspect selected prescriber
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("HCP Profile & Real-Time Context")
    hcp_data = territory_hcps[territory_hcps["hcp_id"] == selected_hcp_id].iloc[0]
    
    st.metric(label="Market Decile", value=int(hcp_data["market_decile"]))
    st.metric(label="Brand Market Share", value=f"{hcp_data['brand_market_share']*100:.1f}%")
    st.metric(label="Claim Rejection Rate", value=f"{hcp_data['claim_rejection_rate']*100:.1f}%")
    st.metric(label="Recent Rep Visits (30d)", value=int(hcp_data["rep_visits_30d"]))
    st.metric(label="Recent Emails Sent (30d)", value=int(hcp_data["emails_received_30d"]))

with col2:
    st.subheader("Next Best Action Recommendation")
    if st.button("Trigger Decision Engine Evaluation"):
        with st.spinner("Executing Causal Uplift & Policy Optimization..."):
            try:
                resp = requests.post(f"{API_URL}/v1/nba/recommend", json={"hcp_id": selected_hcp_id})
                if resp.status_code == 200:
                    result = resp.json()
                    
                    st.success(f"Recommended Action: **{result['recommended_channel']}**")
                    st.info(f"Target Content: **{result['recommended_content']}**")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Predicted Script Uplift", result["predicted_uplift"])
                    m2.metric("Net Expected Utility", result["net_expected_value"])

                    if result["payer_friction_detected"]:
                        st.warning("Payer Friction Flag: High Prior-Authorization or formulary restrictions detected. The policy engine prioritized Access/Reimbursement content.")
                    else:
                        st.info("Payer status clear: Prescriber is eligible for clinical and efficacy detailing.")
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Could not connect to API Gateway at {API_URL}. Ensure app_api.py is running.")