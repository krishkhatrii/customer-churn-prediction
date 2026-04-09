import streamlit as st

st.set_page_config(
    page_title="ChurnGuard Main App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ ChurnGuard: Customer Churn Prediction Suite")
st.markdown("### Welcome to the Churn Prediction Dashboard.")
st.markdown("Use the navigation links in the sidebar to switch between the following modes:")

st.markdown("""
* **Real-Time Dashboard:** Simulates a live data stream to show continuous predictions.
* **Manual Prediction:** Allows you to input custom customer features and receive an immediate prediction.
""")