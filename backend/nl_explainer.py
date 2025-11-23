import numpy as np
import shap

# --- Numeric interpretations for natural language ---
def interpret_age(value):
    if value < -0.5:
        return "a young customer"
    elif value < 0.5:
        return "a middle-aged customer"
    else:
        return "a senior customer"

def interpret_tenure(value):
    if value < -0.5:
        return "a new customer with short tenure"
    elif value < 0.5:
        return "a medium-tenure customer"
    else:
        return "a long-term customer"

def interpret_monthly_charges(value):
    if value < -0.3:
        return "low monthly charges"
    elif value < 0.3:
        return "moderate monthly charges"
    else:
        return "high monthly charges"


# -----------------------------------------------------------
#   MAIN NATURAL LANGUAGE EXPLAINER CLASS
# -----------------------------------------------------------
class NLExplainer:
    def __init__(self, model, shap_explainer, feature_names,
                 gender_map, payment_map, industry_map):
        self.model = model
        self.shap_explainer = shap_explainer
        self.feature_names = feature_names
        self.gender_map = gender_map
        self.payment_map = payment_map
        self.industry_map = industry_map

    def explain(self, feature_vector):
        """
        feature_vector: numpy array of length 6 (gender, paymentmethod, industry, age, tenure, monthlycharges)
        """

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # SHAP values for this single sample (model-agnostic Kernel SHAP)
        shap_vals = self.shap_explainer(feature_vector).values[0]

        # sort SHAP by importance
        top_features = np.argsort(np.abs(shap_vals))[::-1][:5]

        reasons = []

        vector = feature_vector[0]

        for idx in top_features:
            name = self.feature_names[idx]
            raw_value = vector[idx]

            if name == "gender":
                reasons.append(f"is {self.gender_map.get(int(raw_value), 'unknown gender')}")

            elif name == "paymentmethod":
                reasons.append(f"uses {self.payment_map.get(int(raw_value), 'unknown payment method')}")

            elif name == "industry":
                # do not include industry as a reason
                continue

            elif name == "age":
                reasons.append(f"is {interpret_age(raw_value)}")

            elif name == "tenure":
                reasons.append(f"is {interpret_tenure(raw_value)}")

            elif name == "monthlycharges":
                reasons.append(f"has {interpret_monthly_charges(raw_value)}")

            else:
                reasons.append(name)

        if len(reasons) > 1:
            reasons_text = ", ".join(reasons[:-1]) + ", and " + reasons[-1]
        else:
            reasons_text = reasons[0]

        # churn probability
        prob = float(self.model.predict_proba(feature_vector)[0][1])

        # industry intro
        industry_val = int(vector[2])
        industry_name = self.industry_map.get(industry_val, "unknown")

        return (
            f"Customer from the {industry_name} industry is predicted to churn "
            f"with a probability of {prob:.2f}. "
            f"This is mainly because the customer {reasons_text}."
        )
