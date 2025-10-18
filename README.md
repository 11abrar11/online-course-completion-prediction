Online Course Completion Prediction ML Pipeline

An end-to-end machine learning pipeline to predict online course completion, featuring synchronized preprocessing, fixed categorical encoding, a production-ready FastAPI API, and Docker containerization for deployment.


---

🎯 Project Overview

This project predicts whether a student will complete an online course based on engagement metrics, study habits, and demographic features.

Key Highlights:

End-to-end ML pipeline: data preprocessing → model experimentation → training → inference → API service

Accuracy-driven model selection: XGBoost chosen as the best-performing model

Fully synchronized preprocessing for training and inference

Human-readable categorical encoding

Docker-ready for portable deployment



---

🏗️ Project Structure

ML_1/
├── train_model.py              # Script to train XGBoost model with preprocessing
├── inference_model.py          # Class for loading artifacts and making predictions
├── app/
│   └── main.py                 # FastAPI application with prediction endpoints
├── data/                       # Raw and preprocessed datasets
│   ├── online_course_completion.csv
│   ├── preprocessed_online_course_data.csv
│   ├── scaled_train_data.csv
│   └── scaled_test_data.csv
├── models/                     # Saved artifacts (model, scaler, encoders, mappings)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── label_mappings.json
│   └── feature_order.json
├── notebooks/
│   └── Final_File.ipynb        # Data exploration, correlation analysis, and model comparison
├── requirements.txt
├── pyproject.toml              # Poetry dependency management
├── Dockerfile                  # Multi-stage Dockerfile for production deployment
├── docker-compose.yml          # Optional Docker Compose for easier container management
└── README.md


---

🛠️ Workflow

1️⃣ Data Preprocessing & Exploration

Initial analysis, correlation, and feature selection done in the Jupyter notebook

Numerical features scaled, categorical features encoded with preserved mappings

Null values handled consistently to ensure model robustness


2️⃣ Model Training

Three models trained and compared: Logistic Regression, Random Forest, XGBoost

Best model selected based on accuracy (XGBoost)

Training script saves all necessary artifacts for inference (model, scaler, encoders, feature order)


3️⃣ Inference

InferenceModel class loads all artifacts to ensure consistent preprocessing

Accepts new inputs, encodes categorical features, scales numeric features, and predicts completion

Predictions can be decoded to human-readable labels


4️⃣ FastAPI Service

Production-ready API with endpoints:

Root endpoint (/) for welcome message

Health check (/health)

Model info (/model-info)

Prediction (/predict)

Probability predictions (/predict-probabilities)


Input validation ensures robust, reliable API interactions

Swagger UI available for interactive API documentation


5️⃣ Docker Containerization

Multi-stage Dockerfile creates a lightweight, secure production image

Dependencies installed in build stage, runtime image keeps only necessary packages

Non-root user for security

Port exposed for API access

Containerization ensures portability and replicable environments


> Docker enables running the API anywhere without worrying about system dependencies or Python version mismatches. For deployment, it can be run locally, on a cloud instance, or any container orchestration service.




---

🔍 Key Improvements & Solutions

1. Fixed Categorical Encoding: Original category names preserved for interpretability


2. Synchronized Preprocessing: Training and inference pipelines fully consistent


3. Modular Design: Separate scripts for training, inference, and API


4. Production API: FastAPI with validation, documentation, and error handling


5. Artifact Management: All model components versioned and saved


6. Docker-Ready: Multi-stage containerization ensures reproducible deployment




---

🚀 Getting Started

Install dependencies:

Using Poetry or pip


Train the model:

Run the training script; artifacts are saved automatically


Test inference locally:

Use the inference script with new sample inputs


Start FastAPI server:

Run locally to test endpoints and view Swagger documentation


Deploy with Docker (optional):

Build and run the container for a portable production-ready API



---

✅ Project Status

Fully functional FastAPI service with predictions and probability outputs

Synchronized preprocessing ensures reproducible results

Categorical encoding issue resolved (human-readable labels)

Docker-ready for deployment, cloud-ready for future hosting
