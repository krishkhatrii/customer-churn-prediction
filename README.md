# Customer Churn Prediction
# 🧠 Customer Churn Prediction Project

This project aims to predict customer churn (i.e., customers likely to leave a company) using Machine Learning models.  
It follows a modular and scalable structure — designed for teamwork, experimentation, and easy reproducibility.

---

## 📂 Folder Structure Overview
churn-project/
├─ data/ # Datasets (raw and processed)
│ ├─ raw/ # Original datasets (not uploaded to GitHub)
│ └─ processed/ # Cleaned/feature-engineered data
│
├─ notebooks/ # Jupyter notebooks for EDA & experiments
│ ├─ 01_data_exploration.ipynb
│ ├─ 02_feature_engineering.ipynb
│ ├─ 03_model_training.ipynb
│ └─ 04_evaluation.ipynb
│
├─ src/ # Reusable core code (modular Python modules)
│ ├─ data/ # Data loading and cleaning scripts
│ ├─ features/ # Feature engineering and encoding
│ ├─ models/ # Model training, evaluation, and saving
│ └─ explainability/ # Model explainability tools (SHAP, LIME)
│
├─ scripts/ # Command-line runnable scripts
│ ├─ train.py # Train model end-to-end using src code
│ ├─ evaluate.py # Evaluate model and save metrics
│ └─ predict.py # Run predictions on new data
│
├─ tests/ # Unit and integration tests (using pytest)
│ ├─ test_preprocessing.py
│ └─ test_model_training.py
│
├─ models/ # Saved trained models (.pkl / .json)
│
├─ experiments/ # Logs, metrics, plots, and experiment outputs
│
├─ .github/workflows/ # GitHub Actions CI/CD configurations
│
├─ .gitignore # Files and folders excluded from Git tracking
├─ requirements.txt # Python package dependencies
├─ environment.yml # (Optional) Conda environment setup
└─ README.md # Project documentation