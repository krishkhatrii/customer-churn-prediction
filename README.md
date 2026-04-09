# ChurnGuard: Customer Churn Prediction

A Streamlit web app that predicts customer churn in real time using a TDA-enhanced ensemble model, customer segmentation, and natural language SHAP explanations.

---

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app has two pages accessible from the sidebar:

| Page | Description |
|------|-------------|
| **Real-Time Dashboard** | Simulates a live data stream — auto-refreshes every 10 seconds with a randomly generated customer and prediction |
| **Manual Prediction** | Enter customer details manually and get an instant churn forecast |

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── streamlit_app.py              # App entry point, sidebar navigation
│
├── pages/
│   ├── Real_Time_Dashboard.py    # Real-time simulation page
│   └── Manual_Prediction.py     # Manual input prediction page
│
├── backend/
│   ├── realtime.py               # Core prediction engine (preprocessing, model inference, SHAP)
│   ├── nl_explainer.py           # NLExplainer class (archived, replaced by inline logic)
│   └── artifacts/                # Trained model files loaded at runtime
│       ├── super_ensemble.pkl        # Ensemble churn model (XGBoost + LightGBM + CatBoost)
│       ├── scaler.pkl                # Numeric feature scaler
│       ├── mappings.pkl              # Label encoders for categorical features
│       ├── tda_node_centers.npy      # TDA topology node centers
│       ├── tda_feature_columns.pkl   # 350 feature column names (base + TDA one-hot)
│       ├── segmentation_model.pkl    # KMeans customer segmentation model
│       ├── kmeans_scaler.pkl         # Scaler for segmentation features
│       ├── kmeans_encoder.pkl        # One-hot encoder for segmentation
│       └── kmeans_feature_columns.pkl
│
├── data/
│   ├── raw/                          # Original source datasets (3 industries)
│   │   ├── Telco-Customer-Churn.csv
│   │   ├── Subscription_Service_Churn_Dataset.csv
│   │   └── ecommerce_transactions.csv
│   ├── combined_cleaned_encoded.csv  # Cleaned + encoded combined dataset (model training input)
│   ├── combined_cleaned_unencoded.csv# Cleaned dataset before encoding
│   ├── customer_features.csv         # 6-feature subset (segmentation + encoder input)
│   └── train_test_data/              # Train/test splits produced by model_training.ipynb
│       ├── X_train.csv / y_train.csv
│       ├── X_train_smote.csv / y_train_smote.csv   # SMOTE-balanced training set
│       ├── X_test.csv / y_test.csv
│
└── notebooks/                        # Full training pipeline
    ├── data_preprocessing.ipynb      # 1. Combines raw CSVs, cleans and encodes
    ├── feature_extraction.ipynb      # 2. Extracts 6-feature subset
    ├── model_training.ipynb          # 3. TDA feature engineering + ensemble training
    ├── save_label_mappings.ipynb     # 4. Saves categorical label encoders
    ├── segmentation_and_clv.ipynb    # 5. KMeans segmentation + CLV analysis
    ├── explainability_analysis.ipynb # SHAP + LIME analysis (reference, not required for app)
```

---

## 🧠 How It Works

Each prediction runs through 4 steps in `backend/realtime.py`:

1. **Encoding** — Categorical inputs (gender, payment method, industry) are label-encoded using `mappings.pkl`. Numeric inputs (age, tenure, monthly charges) are scaled.
2. **TDA features** — The encoded vector is assigned to its nearest TDA node (Topological Data Analysis), producing a 350-dimensional feature vector.
3. **Prediction** — The ensemble model outputs a churn probability and binary prediction.
4. **Segmentation & CLV** — A separate KMeans model assigns the customer to a value segment. CLV is estimated as `monthly charges × tenure × 1.2`. Both signals are combined to produce a final customer value label (Low / Medium / High).
5. **Explanation** — SHAP values identify the top contributing features, which are converted into a natural language sentence.

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| App framework | Streamlit |
| ML models | XGBoost, LightGBM, CatBoost |
| Ensemble & preprocessing | scikit-learn |
| Class imbalance | imbalanced-learn (SMOTE) |
| Topological Data Analysis | KeplerMapper |
| Explainability | SHAP, LIME |
| Data | pandas, numpy |
