import streamlit as st
import time
import random
from backend.realtime import predict_realtime

# ---------------------------
# Generate Random Incoming Data
# ---------------------------
def generate_random_customer():
    return {
        "gender": random.choice(["Male", "Female"]),
        "paymentmethod": random.choice(["UPI", "Credit Card", "Credit card (automatic)", "Debit Card", "Net Banking", "PayPal", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Cash on Delivery" ]),
        "industry": random.choice(["Ecommerce", "Subscription", "Telecom"]),
        "age": random.randint(20, 70),
        "tenure": random.randint(1, 60),
        "monthlycharges": random.randint(20, 130)
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Real-Time Customer Churn Prediction", layout="wide")

st.title("Real-Time Customer Churn Prediction Dashboard")
st.write("This dashboard simulates live incoming customer data and displays model predictions in real-time.")

placeholder = st.empty()

while True:
    with placeholder.container():
        # Generate simulated incoming data
        incoming = generate_random_customer()

        st.subheader("Incoming Customer Data")
        st.json(incoming)

        # Predict using backend
        output = predict_realtime(incoming)

        # Display Model Output
        st.subheader("Model Output")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Prediction", "Yes" if output["prediction"] == 1 else "No")
            st.metric("Churn Probability", f"{output['probability']:.2f}")

        with col2:
            st.metric("Customer Segment", output["segment"])
            st.metric("CLV Estimate", f"{output['clv']:.0f}")

        with col3:
            st.write("### Recommended Action")
            st.success(output["recommended_action"])

        st.write("### Explanation")
        st.info(output["explanation"])

    time.sleep(4)   # refresh every 4 seconds
