"""
Description
"""

# Imports
import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Demand Forecasting", layout="wide", initial_sidebar_state="expanded"
)
# API Configuration
API_BASE_URL = "http://localhost:8000"
# Styling
st.markdown(
    """
<style>
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except Exception:
        return False


# main
st.markdown(
    """
    <div class="header">
        <h1>Demand Forecasting Dashboard</h1>
        <p>Predict hourly demand using XGBoost time series model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Check API health
if not check_api_health():
    st.error("**API is not running!**")
    st.stop()

st.success("API is running")
