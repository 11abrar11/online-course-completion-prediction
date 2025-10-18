"""
FastAPI Application for Online Course Completion Prediction

This FastAPI application provides REST API endpoints for the machine learning model
that predicts whether a student will complete an online course.

Endpoints:
- GET /: Welcome message and API information
- POST /predict: Make a prediction on student data
- GET /model-info: Get information about the model and features
- GET /health: Health check endpoint

Features:
- Input validation using Pydantic models
- Comprehensive error handling
- Detailed API documentation with Swagger UI
- Support for both prediction and probability endpoints
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Union
import uvicorn
import sys
import os

# Add the parent directory to the path to import our inference model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_model import InferenceModel


# Initialize FastAPI application
app = FastAPI(
    title="Online Course Completion Prediction API",
    description="""
    This API provides machine learning predictions for online course completion.
    
    ## Features
    
    * Predict whether a student will complete an online course
    * Get prediction probabilities
    * Input validation and error handling
    * Comprehensive API documentation
    
    ## Model Information
    
    The model uses XGBoost and considers the following features:
    - Country of origin
    - Hours per week spent on the course
    - Number of logins in the last month
    - Percentage of videos watched
    - Number of assignments submitted
    - Number of discussion posts
    - Whether the student is a working professional
    - Preferred device for learning
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize the inference model
try:
    inference_model = InferenceModel(model_dir="models/")
    model_loaded = True
    print("✓ Inference model loaded successfully")
except Exception as e:
    model_loaded = False
    print(f"✗ Failed to load inference model: {e}")


# Pydantic models for request/response validation
class StudentData(BaseModel):
    """
    Pydantic model for student input data validation.
    
    This model ensures that all required features are provided
    and validates their data types and ranges.
    """
    country: str = Field(
        ..., 
        description="Country of origin (e.g., 'India', 'USA', 'Brazil')",
        example="India"
    )
    hours_per_week: float = Field(
        ..., 
        ge=0, 
        le=168,  # Maximum hours in a week
        description="Hours per week spent on the course",
        example=10.5
    )
    num_logins_last_month: int = Field(
        ..., 
        ge=0, 
        description="Number of logins in the last month",
        example=15
    )
    videos_watched_pct: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Percentage of videos watched (0-100)",
        example=75.0
    )
    assignments_submitted: int = Field(
        ..., 
        ge=0, 
        description="Number of assignments submitted",
        example=5
    )
    discussion_posts: int = Field(
        ..., 
        ge=0, 
        description="Number of discussion posts made",
        example=3
    )
    is_working_professional: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Whether the student is a working professional (0 or 1)",
        example=1
    )
    preferred_device: str = Field(
        ..., 
        description="Preferred device for learning (e.g., 'mobile', 'desktop', 'tablet')",
        example="mobile"
    )

    @validator('country')
    def validate_country(cls, v):
        """Validate that country is not empty."""
        if not v.strip():
            raise ValueError('Country cannot be empty')
        return v.strip()

    @validator('preferred_device')
    def validate_preferred_device(cls, v):
        """Validate that preferred device is not empty."""
        if not v.strip():
            raise ValueError('Preferred device cannot be empty')
        return v.strip()


class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response.
    """
    prediction: int = Field(..., description="Prediction (0 = Not Completed, 1 = Completed)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    input_data: StudentData = Field(..., description="Original input data")


class ModelInfoResponse(BaseModel):
    """
    Pydantic model for model information response.
    """
    model_type: str = Field(..., description="Type of machine learning model")
    num_features: int = Field(..., description="Number of input features")
    feature_order: list = Field(..., description="Order of features expected by the model")
    categorical_mappings: Dict[str, Dict[str, int]] = Field(..., description="Categorical value mappings")


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint that provides welcome message and API information.
    
    Returns:
        Dict: Welcome message and basic API information
    """
    return {
        "message": "Welcome to the Online Course Completion Prediction API",
        "description": "This API predicts whether a student will complete an online course",
        "version": "1.0.0",
        "docs": "/docs",
        "model_info": "/model-info"
    }


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model and its features.
    
    Returns:
        ModelInfoResponse: Model information including features and mappings
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model artifacts."
        )
    
    try:
        model_info = inference_model.get_feature_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(student_data: StudentData):
    """
    Make a prediction on student data.
    
    This endpoint accepts student information and returns a prediction
    about whether they will complete the online course.
    
    Args:
        student_data (StudentData): Student information including all required features
        
    Returns:
        PredictionResponse: Prediction result with probabilities and input data
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model artifacts."
        )
    
    try:
        # Convert Pydantic model to dictionary
        input_dict = student_data.dict()
        
        # Make prediction
        prediction = inference_model.predict(input_dict)
        probabilities = inference_model.predict_proba(input_dict)
        
        # Get human-readable prediction
        prediction_label = inference_model.decode_prediction(prediction)
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            probabilities=probabilities,
            input_data=student_data
        )
        
    except ValueError as e:
        # Handle validation errors (e.g., unseen categorical values)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation error: {str(e)}"
        )
    except Exception as e:
        # Handle other prediction errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-probabilities")
async def predict_probabilities(student_data: StudentData):
    """
    Get only the prediction probabilities without the full prediction.
    
    This endpoint is useful when you only need the probability scores
    without the binary prediction.
    
    Args:
        student_data (StudentData): Student information
        
    Returns:
        Dict: Only the probability scores
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model artifacts."
        )
    
    try:
        input_dict = student_data.dict()
        probabilities = inference_model.predict_proba(input_dict)
        
        return {
            "probabilities": probabilities,
            "input_data": student_data.dict()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Probability prediction failed: {str(e)}"
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )


def main():
    """
    Main function to run the FastAPI application.
    """
    print("Starting Online Course Completion Prediction API...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("Health check available at: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()

