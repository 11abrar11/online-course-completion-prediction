# Online Course Completion Prediction

This project aims to predict whether a student will complete an online course using a machine learning model trained on behavioral and demographic features.

## 🔍 Problem Statement

The goal is to build a classification model that can predict the value of the `completed_course` column (binary: 0 or 1) based on various input features.

## 🧰 Tech Stack

- **Python**
- **Jupyter Notebook**
- **Pandas, NumPy** (for data manipulation)
- **scikit-learn** (for ML models and preprocessing)
- **XGBoost** (for advanced boosting-based model)
- **Matplotlib / Seaborn** (for visualizations)
- **Poetry** (for environment and dependency management)

## 📊 Data Preprocessing

- Feature selection using correlation and categorical distribution analysis
- Missing values handled using mean/mode imputation
- Categorical variables encoded using One-Hot Encoding
- Feature scaling using StandardScaler (Z-score normalization)

## 🧠 Model Training

Three models were trained and compared:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Performance evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 📁 Files in This Repository

- `notebooks/`: Contains Jupyter notebooks used for data exploration and training


## ✍️ Author

Mohammed Abrar Hussain

---
