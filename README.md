# Customer Churn Prediction
# 🧠 Customer Churn Prediction Project

This project aims to predict customer churn (i.e., customers likely to leave a company) using Machine Learning models.  
It follows a modular and scalable structure — designed for teamwork, experimentation, and easy reproducibility.

---

## 📁 Project Folder Structure

**Root directory:** `customer-churn-prediction/`

### 🗂️ 1. `data/`
- **Purpose:** Store datasets used in the project.
- **Subfolders:**
  - `raw/` – Original unprocessed datasets (not pushed to GitHub).
  - `processed/` – Cleaned and transformed data ready for modeling.
- **Usage example:**
  - Input: `data/raw/churn.csv`
  - Output: `data/processed/churn_clean.csv`

---

### 📒 2. `notebooks/`
- **Purpose:** Jupyter notebooks for exploration and experiments.
- **Example notebooks:**
  - `01_data_exploration.ipynb` – Explore and visualize churn patterns.
  - `02_feature_engineering.ipynb` – Create and encode new features.
  - `03_model_training.ipynb` – Train and tune ML models.
  - `04_evaluation.ipynb` – Evaluate model performance.
- **Tip:** Move finalized, reusable code from notebooks into `src/`.

---

### ⚙️ 3. `src/`
- **Purpose:** Core reusable source code (modular Python scripts).
- **Subfolders:**
  - `data/` – Data loading and cleaning functions.
  - `features/` – Feature transformation and encoding.
  - `models/` – Model training, evaluation, and saving.
  - `explainability/` – Model interpretation tools (e.g., SHAP, LIME).
- **Usage:**  
  Import reusable functions like:
  ```python
  from src.models.train_model import train_rf_model
