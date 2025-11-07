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

---

### 🧩 4. `scripts/`

- **Purpose:** Command-line scripts to run pipelines end-to-end.  
- **Files:**
  - `train.py` – Train models using code from `src/`.
  - `evaluate.py` – Evaluate trained models and log metrics.
  - `predict.py` – Generate predictions on new data.
- **Usage:**
  ```bash
  python scripts/train.py

---

### 🧪 5. `tests/`

- **Purpose:** Contains automated tests for your code.  
- **Examples:**
  - `test_preprocessing.py`
  - `test_model_training.py`
- **Usage:**
  ```bash
  pytest

---

### 🤖 6. `.github/workflows/`

- **Purpose:** Holds GitHub Actions (CI/CD) configuration files.  
- **Usage:** Automate testing or code formatting checks on every pull request.

---

### 🧠 7. `models/`

- **Purpose:** Store trained model files (e.g., `.pkl`, `.joblib`, `.json`).  
- **Tip:** Add this folder to `.gitignore` to avoid pushing large binaries.

---

### 📊 8. `experiments/`

- **Purpose:** Store experiment outputs, logs, metrics, and plots.  
- **Example structure:**
experiments/
├─ run_001/
│  ├─ metrics.json
│  ├─ confusion_matrix.png
│  └─ model.pkl

---

🧾 9. Other important files
- .gitignore	->  Tells Git which files/folders to ignore (e.g., data, models).
- requirements.txt  ->  Lists all Python dependencies.
- environment.yml	->  (Optional) Conda environment setup file.
- README.md	->  Documentation for your project.

---


## ⚙️ How to Use This Structure in Practice
| Stage                   | Work in                                                  | Save output in            | Move stable code to                   |
| ----------------------- | -------------------------------------------------------- | ------------------------- | ------------------------------------- |
| **Data cleaning**       | `notebooks/01_data_exploration.ipynb`                    | `data/processed/`         | `src/data/preprocess.py`              |
| **Feature engineering** | `notebooks/02_feature_engineering.ipynb`                 | `data/processed/`         | `src/features/feature_engineering.py` |
| **Model training**      | `scripts/train.py`                                       | `models/`, `experiments/` | `src/models/train_model.py`           |
| **Evaluation**          | `scripts/evaluate.py` or `notebooks/04_evaluation.ipynb` | `experiments/`            | `src/models/evaluate.py`              |
| **Explainability**      | `notebooks/05_explainability.ipynb`                      | `experiments/`            | `src/explainability/`                 |

