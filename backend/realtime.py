# backend/realtime.py
import joblib
import numpy as np
import pandas as pd
import os
from .nl_explainer import NLExplainer

# -------------------------
# Load backend artifacts
# -------------------------
artifact_path = "backend/artifacts/"

model = joblib.load(artifact_path + "super_ensemble.pkl")
scaler = joblib.load(artifact_path + "scaler.pkl")
mappings = joblib.load(artifact_path + "mappings.pkl")            # dict of label encoders
tda_centers = np.load(artifact_path + "tda_node_centers_1.npy", allow_pickle=True)
feature_columns = joblib.load(os.path.join(artifact_path, "feature_columns_1.pkl"))  # list of final columns for TDA (length 350)

kmeans_scaler = joblib.load(artifact_path + "kmeans_scaler.pkl")
kmeans_encoder = joblib.load(artifact_path + "kmeans_encoder.pkl")
kmeans_cols = joblib.load(artifact_path + "kmeans_feature_columns.pkl")
segmenter = joblib.load(artifact_path + "segmentation_model.pkl")

nl_explainer = joblib.load(artifact_path + "nl_explainer.pkl")    # callable for natural language explanation
shap_bundle = joblib.load(artifact_path + "shap_explainer.pkl")
shap_explainer = shap_bundle["explainer"]
shap_feature_names = joblib.load(artifact_path + "xai_feature_names.pkl")

# From mappings (encoder values)
gender_map = {v: k for k, v in mappings["gender"].items()}
payment_map = {v: k for k, v in mappings["paymentmethod"].items()}
industry_map = {v: k for k, v in mappings["industry"].items()}

nl_explainer = NLExplainer(
    model=model,
    shap_explainer=shap_explainer,
    feature_names=feature_names,
    gender_map=gender_map,
    payment_map=payment_map,
    industry_map=industry_map
)


# -------------------------
# Helper functions
# -------------------------
def encode_row(input_dict):
    encoded = []

    for col, mapping in mappings.items():
        value = input_dict[col]

        # unseen → -1
        encoded_value = mapping.get(value, -1)
        encoded.append(encoded_value)

    # append numeric
    for col in ["age", "tenure", "monthlycharges"]:
        encoded.append(float(input_dict[col]))

    return np.array(encoded, dtype=float)

def assign_tda_node(base_vector):
    distances = np.linalg.norm(tda_centers - base_vector, axis=1)
    return int(np.argmin(distances))

def preprocess_realtime(input_dict):
    # -------------------------------
    # 1. MAIN MODEL ENCODING (350 features)
    # -------------------------------
    encoded_and_numeric = encode_row(input_dict)

    n_cats = len(mappings)
    encoded_cats = encoded_and_numeric[:n_cats]
    numeric_vals = encoded_and_numeric[n_cats:]

    scaled_numeric = scaler.transform([numeric_vals])[0]

    base_for_tda = np.concatenate([encoded_cats, scaled_numeric])
    tda_node_index = assign_tda_node(base_for_tda)

    tda_onehot = np.zeros(len(tda_centers))
    tda_onehot[tda_node_index] = 1

    # final 350 features
    final_features = np.concatenate([encoded_cats, scaled_numeric, tda_onehot])

    # sanity
    if final_features.shape[0] != len(feature_columns):
        raise ValueError(
            f"Feature mismatch: got {final_features.shape[0]}, expected {len(feature_columns)}"
        )

    # -------------------------------
    # 2. FEATURES FOR KMEANS (18 features)
    # -------------------------------
    kmeans_cat_cols = ['gender', 'paymentmethod', 'industry']

    kmeans_num_array = kmeans_scaler.transform(
        [[input_dict['age'], input_dict['tenure'], input_dict['monthlycharges']]]
    )[0]

    cat_array_for_kmeans = kmeans_encoder.transform(
        [[input_dict[c] for c in kmeans_cat_cols]]
    ).flatten()

    features_kmeans = np.concatenate([cat_array_for_kmeans, kmeans_num_array])

    # -------------------------------
    # 3. RAW EXPLAINER VECTOR (6 features)
    # -------------------------------
    explainer_vector = np.array([
        input_dict["gender"],
        input_dict["paymentmethod"],
        input_dict["industry"],
        float(input_dict["age"]),
        float(input_dict["tenure"]),
        float(input_dict["monthlycharges"]),
    ])

    return final_features, features_kmeans, explainer_vector


# Main prediction function
def predict_realtime(input_dict):

    features_350, features_kmeans, explainer_vector = preprocess_realtime(input_dict)

    # model prediction
    probability = float(model.predict_proba([features_350])[0][1])
    prediction = int(model.predict([features_350])[0])

    # segmentation
    segment = int(segmenter.predict([features_kmeans])[0])

    # CLV
    monthly = float(input_dict.get("monthlycharges", 0))
    tenure = float(input_dict.get("tenure", 0))
    clv_value = monthly * tenure * 1.2

    # explanation from natural language explainer
    explanation = nl_explainer.explain(explainer_vector)

    # Recommended action
    if probability >= 0.7 and clv_value > 2000:
        action = "Offer strong discount"
    elif probability >= 0.7:
        action = "Send retention SMS"
    elif probability >= 0.4:
        action = "Engagement email"
    else:
        action = "No action needed"

    return {
        "prediction": prediction,
        "probability": probability,
        "segment": segment,
        "clv": clv_value,
        "explanation": explanation,
        "recommended_action": action
    }

