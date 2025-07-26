## 🔧 Tools & Libraries

- Python
- Jupyter Notebook
- *Poetry* for dependency management
- pandas, numpy for data handling
- matplotlib, seaborn for visualization
- scikit-learn for preprocessing and ML models
- xgboost for gradient boosting classifier

---

## 🔄 Workflow Summary

### 1. Data Preprocessing

- Loaded dataset using pandas
- Identified and handled null values
- Performed feature selection using correlation and Cramér's V
- Applied one-hot encoding for categorical variables
- Normalized features using StandardScaler

### 2. Feature Engineering

- Handled missing values (mean/mode imputation)
- Converted categorical columns into numerical format
- Final dataset with selected 9 features

### 3. Model Training

Trained the following models:
- *Logistic Regression*
- *Random Forest*
- *XGBoost*

All models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 4. Evaluation

Metrics were printed and visualized using seaborn heatmaps.

---

## ✅ Result

All models were successfully trained. Evaluation metrics were printed for each. The model with the best performance can be further fine-tuned or deployed if needed.

---

## 🚀 Future Work

- Perform hyperparameter tuning using GridSearchCV
- Save model using pickle or joblib
- Deploy the model using Flask or FastAPI

---

## 🙋‍♂️ Author

Mohammed Abrar Hussain  
ML Engineering Beginner | Guided by Mentor

---

## 📌 Note

This project was built as part of a mentor-guided learning task using only Jupyter Notebook and standard ML libraries. No prior experience in ML or Python was assumed during the build.
