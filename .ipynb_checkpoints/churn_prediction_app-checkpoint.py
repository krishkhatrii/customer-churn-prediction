import streamlit as st
import time
import random
# NOTE: Removed sys and os imports, as they are no longer needed
# because the app runs from the root and can access 'backend' directly.
from backend.realtime import predict_realtime 

# ----------------------------------------------------------------------
# GLOBAL FUNCTIONS (Shared by both views)
# ----------------------------------------------------------------------

def generate_random_customer():
    """Generates a random set of customer features for simulation."""
    return {
        "gender": random.choice(["Male", "Female"]),
        "paymentmethod": random.choice(["UPI", "Credit Card", "Credit card (automatic)", "Debit Card", "Net Banking", "PayPal", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Cash on Delivery" ]),
        "industry": random.choice(["Ecommerce", "Subscription", "Telecom"]),
        "age": random.randint(20, 70),
        "tenure": random.randint(1, 60),
        "monthlycharges": random.randint(20, 130) 
    }

def display_detail(label, value, col):
    """Displays a label and value inside a styled Streamlit container column, mimicking a textbox look."""
    with col:
        st.markdown(f"**{label}:**")
        st.code(str(value), language='text')

def display_results(output):
    """Displays the model output using the styled metrics and boxes (used by both pages)."""
    
    # Determine visual style based on prediction
    is_churn = output["prediction"] == 1
    churn_status = "Yes" if is_churn else "No"

    st.header("📊 Prediction Results")

    # Using 4 columns for the key metrics for a clean, spaced-out look
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Churn Prediction", 
            value=churn_status,
            delta="HIGH RISK" if is_churn else "LOW RISK",
            delta_color="inverse" if is_churn else "normal"
        )

    with metric_col2:
        st.metric(
            label="Churn Probability", 
            value=f"{output['probability']:.2f}",
        )

    with metric_col3:
        st.metric("Customer Segment", value=output["segment"]) 

    with metric_col4:
        st.metric(
            label="CLV Estimate", 
            value=f"${output['clv']:.0f}"
        )

    # Recommended Action
    st.markdown("### Recommended Action:")
    
    action_text = output["recommended_action"]
    
    if is_churn:
        st.warning(action_text)
    else:
        st.success(action_text)

    # Model Explanation
    st.markdown("### Model Explanation:") 
    st.info(output["shap_explanation"])

# ----------------------------------------------------------------------
# PAGE FUNCTIONS
# ----------------------------------------------------------------------

def real_time_dashboard():
    """Renders the continuously updating real-time dashboard view."""
    st.title("🛡️ ChurnGuard: Real-Time Dashboard")
    st.markdown("### *Predict. Prevent. Retain.*")

    st.markdown("", unsafe_allow_html=True) # Space
    st.markdown("", unsafe_allow_html=True) # Space

    placeholder = st.empty()

    # NOTE: The endless loop is used here to simulate real-time updates.
    while True:
        with placeholder.container():
            # Generate simulated incoming data
            incoming = generate_random_customer()

            st.header("👤 Incoming Customer Details")
            
            col_a, col_b = st.columns(2)

            # Display details
            display_detail("Gender", incoming['gender'], col_a)
            display_detail("Payment Method", incoming['paymentmethod'], col_a)
            display_detail("Industry", incoming['industry'], col_a)
            display_detail("Age (Years)", incoming['age'], col_b)
            display_detail("Tenure (Months)", incoming['tenure'], col_b)
            display_detail("Monthly Charges ($)", f"${incoming['monthlycharges']:.2f}", col_b)
            
            st.markdown("---") 

            # Predict and display
            try:
                output = predict_realtime(incoming)
                display_results(output)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            
        time.sleep(10)


def manual_prediction_form():
    """Renders the user input form for manual prediction."""
    st.title("✍️ Manual Customer Prediction")
    st.markdown("### *Predict. Prevent. Retain.*")
    st.markdown("Enter the customer details below and click 'Predict' to get an instant churn forecast.")

    # Use session state to hold prediction results without rerunning the full app
    if 'prediction_output' not in st.session_state:
        st.session_state.prediction_output = None
        st.session_state.submitted = False

    # Create the form for input
    with st.form(key='churn_input_form'):
        st.subheader("Customer Features")
        col1, col2 = st.columns(2)

        # Left Column Inputs
        gender = col1.radio("Gender", ["Male", "Female"])
        industry = col1.selectbox("Industry", ["Ecommerce", "Subscription", "Telecom"])
        age = col1.slider("Age (Years)", 18, 99, 45)

        # Right Column Inputs
        paymentmethod = col2.selectbox("Payment Method", [
            "UPI", "Credit Card", "Credit card (automatic)", "Debit Card", "Net Banking", 
            "PayPal", "Electronic check", "Mailed check", "Bank transfer (automatic)", 
            "Cash on Delivery"
        ])
        tenure = col2.slider("Tenure (Months)", 0, 72, 24)
        monthlycharges = col2.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=65.0, step=1.0)

        # Submit Button
        submitted = st.form_submit_button("Run Prediction")
        
        if submitted:
            input_data = {
                "gender": gender,
                "paymentmethod": paymentmethod,
                "industry": industry,
                "age": age,
                "tenure": tenure,
                "monthlycharges": monthlycharges
            }

            try:
                with st.spinner('Calculating Churn Prediction...'):
                    prediction_output = predict_realtime(input_data)
                
                # Store results in session state
                st.session_state.prediction_output = prediction_output
                st.session_state.submitted = True
                # Rerun to clear the form and display results
                st.rerun() 
                
            except Exception as e:
                st.error(f"An error occurred during prediction. Please check the backend connection: {e}")
                st.session_state.submitted = False

    # Display results outside the form after submission
    if st.session_state.submitted and st.session_state.prediction_output:
        display_results(st.session_state.prediction_output)

# ----------------------------------------------------------------------
# MAIN APP NAVIGATION
# ----------------------------------------------------------------------

# Set page configuration once
st.set_page_config(page_title="ChurnGuard Suite", layout="wide")

# Sidebar for navigation control
st.sidebar.title("App Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Real-Time Dashboard", "Manual Prediction"]
)

# Conditional rendering based on selection
if page_selection == "Real-Time Dashboard":
    real_time_dashboard()
elif page_selection == "Manual Prediction":
    manual_prediction_form()