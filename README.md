Online Course Completion Prediction

This project predicts whether a student will complete an online course using a machine learning model trained on behavioral and demographic features.


---

🔍 Problem Statement

The goal is to build a classification model that predicts the completed_course column (binary: 0 or 1) based on student activity, engagement, and demographic data.


---

🧰 Tech Stack

Python 3.12

Jupyter Notebook (for exploration, preprocessing, and model experimentation)

Pandas & NumPy (data manipulation)

scikit-learn (preprocessing, model evaluation)

XGBoost (best-performing ML model)

Matplotlib & Seaborn (visualizations)

Poetry (environment and dependency management)

FastAPI (REST API for model inference)

Docker (containerization for deployment)



---

📊 Data Preprocessing

Feature selection using correlation matrices and categorical distribution analysis

Missing values handled consistently across numerical and categorical features

Categorical features encoded with human-readable mappings to maintain interpretability

Feature scaling applied to numeric features for consistent model input



---

🧠 Model Training

Three models trained and compared:

Logistic Regression

Random Forest Classifier

XGBoost Classifier


Evaluation metrics: Accuracy (primary), Precision, Recall, F1 Score, Confusion Matrix

XGBoost selected as best model based on highest accuracy

Training script saves model, encoders, scaler, and feature order for reproducible inference



---

⚡ FastAPI Service

REST API endpoints:

/ → Welcome message

/health → Health check

/model-info → Feature mappings and model info

/predict → Make a prediction

/predict-probabilities → Get probability scores


Input validation ensures data integrity

Swagger UI documentation available for interactive testing



---

🗂️ Repository Structure

ML_1/
├── train_model.py              # Model training script
├── inference_model.py          # Model inference class
├── app/
│   └── main.py                 # FastAPI application
├── data/                       # Raw and preprocessed datasets
├── models/                     # Saved artifacts (model, scaler, encoders)
├── notebooks/                  # Jupyter notebooks for analysis and training
├── requirements.txt
├── pyproject.toml              # Poetry dependencies
├── Dockerfile                  # Multi-stage Dockerfile for production
├── docker-compose.yml          # Optional Docker Compose
└── README.md


---

🐳 Docker Containerization

Multi-stage Dockerfile creates a production-ready image with only necessary dependencies

Non-root user ensures security best practices

Container exposes port 8000 for API access

FastAPI runs inside container exactly as on local environment

Docker provides portability, reproducibility, and easier deployment on any server or cloud platform


Deployment Workflow:

1. Build Docker image


2. Run container on any host or cloud service


3. API accessible via host port, with Swagger docs and health check




---

✅ Key Features

Fixed categorical encoding with JSON mappings for human-readable inference

Synchronized preprocessing between training and API inference

Accuracy-driven model selection

Production-ready FastAPI with input validation, error handling, and documentation

Dockerized for portable, reproducible deployment



---

✍️ Author

Mohammed Abrar Hussain
