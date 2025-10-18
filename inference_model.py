"""
Machine Learning Model Inference Script

This script handles model inference for the online course completion prediction model.
It loads the trained model and all preprocessing artifacts to make predictions on new data.

Key Features:
- Loads trained model, scaler, and encoders
- Handles categorical encoding using saved mappings
- Provides prediction and probability methods
- Includes input validation and error handling
"""

import pickle
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional


class InferenceModel:
    """
    Inference class for the online course completion prediction model.
    
    This class handles:
    - Loading trained model and preprocessing artifacts
    - Categorical encoding using saved mappings
    - Feature scaling and transformation
    - Making predictions on new data
    - Input validation and error handling
    """
    
    def __init__(self, model_dir: str = "models/"):
        """
        Initialize the inference model by loading all required artifacts.
        
        Args:
            model_dir (str): Directory containing model artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.encoder_mappings = {}
        self.feature_order = None
        
        # Load all artifacts
        self._load_artifacts()
        
        print("InferenceModel initialized successfully!")
        print(f"Model directory: {self.model_dir}")
        print(f"Features: {self.feature_order}")
        print(f"Categorical mappings available for: {list(self.encoder_mappings.keys())}")

    def _load_artifacts(self) -> None:
        """
        Load all model artifacts from the model directory.
        
        Loads:
        - Trained model (best_model.pkl)
        - Feature scaler (scaler.pkl)
        - Categorical encoders (encoders.pkl)
        - Encoder mappings (label_mappings.json)
        - Feature order (feature_order.json)
        """
        print("Loading model artifacts...")
        
        try:
            # Load the trained model
            model_path = os.path.join(self.model_dir, "best_model.pkl")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from: {model_path}")
            
            # Load the scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded from: {scaler_path}")
            
            # Load the encoders
            encoders_path = os.path.join(self.model_dir, "encoders.pkl")
            with open(encoders_path, "rb") as f:
                self.encoders = pickle.load(f)
            print(f"✓ Encoders loaded from: {encoders_path}")
            
            # Load encoder mappings
            mappings_path = os.path.join(self.model_dir, "label_mappings.json")
            with open(mappings_path, "r") as f:
                self.encoder_mappings = json.load(f)
            print(f"✓ Encoder mappings loaded from: {mappings_path}")
            
            # Load feature order
            feature_order_path = os.path.join(self.model_dir, "feature_order.json")
            with open(feature_order_path, "r") as f:
                self.feature_order = json.load(f)
            print(f"✓ Feature order loaded from: {feature_order_path}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required artifact not found: {e}. Please run training first.")
        except Exception as e:
            raise Exception(f"Error loading artifacts: {e}")

    def _validate_input(self, input_data: Dict[str, Union[str, int, float]]) -> None:
        """
        Validate input data format and required fields.
        
        Args:
            input_data (Dict): Input data dictionary
            
        Raises:
            ValueError: If input validation fails
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        # Check if all required features are present
        missing_features = set(self.feature_order) - set(input_data.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for extra features
        extra_features = set(input_data.keys()) - set(self.feature_order)
        if extra_features:
            print(f"Warning: Extra features provided (will be ignored): {extra_features}")

    def _encode_categorical_features(self, input_data: Dict[str, Union[str, int, float]]) -> Dict[str, Union[int, float]]:
        """
        Encode categorical features using saved mappings.
        
        Args:
            input_data (Dict): Input data with categorical values
            
        Returns:
            Dict: Input data with encoded categorical values
        """
        encoded_data = input_data.copy()
        
        # Encode each categorical feature
        for feature, mapping in self.encoder_mappings.items():
            if feature in encoded_data:
                original_value = str(encoded_data[feature])
                
                # Check if the value exists in the mapping
                if original_value in mapping:
                    encoded_data[feature] = mapping[original_value]
                else:
                    # Handle unseen categories by using the most common encoding (0)
                    print(f"Warning: Unseen category '{original_value}' for feature '{feature}'. Using default encoding.")
                    encoded_data[feature] = 0
        
        return encoded_data

    def _prepare_features(self, input_data: Dict[str, Union[str, int, float]]) -> np.ndarray:
        """
        Prepare features for prediction by encoding and scaling.
        
        Args:
            input_data (Dict): Raw input data
            
        Returns:
            np.ndarray: Prepared feature array
        """
        # Validate input
        self._validate_input(input_data)
        
        # Encode categorical features
        encoded_data = self._encode_categorical_features(input_data)
        
        # Create DataFrame with correct feature order
        feature_values = [encoded_data[feature] for feature in self.feature_order]
        feature_array = np.array([feature_values])
        
        # Scale features using the saved scaler
        scaled_features = self.scaler.transform(feature_array)
        
        return scaled_features

    def predict(self, input_data: Dict[str, Union[str, int, float]]) -> int:
        """
        Make a prediction on input data.
        
        Args:
            input_data (Dict): Input data dictionary with feature values
            
        Returns:
            int: Prediction (0 or 1)
        """
        try:
            # Prepare features
            features = self._prepare_features(input_data)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            return int(prediction)
            
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")

    def predict_proba(self, input_data: Dict[str, Union[str, int, float]]) -> Dict[str, float]:
        """
        Get prediction probabilities for input data.
        
        Args:
            input_data (Dict): Input data dictionary with feature values
            
        Returns:
            Dict: Dictionary with class probabilities
        """
        try:
            # Prepare features
            features = self._prepare_features(input_data)
            
            # Get probabilities
            probabilities = self.model.predict_proba(features)[0]
            
            return {
                "not_completed": float(probabilities[0]),
                "completed": float(probabilities[1])
            }
            
        except Exception as e:
            raise Exception(f"Probability prediction failed: {e}")

    def get_feature_info(self) -> Dict[str, any]:
        """
        Get information about the model features and mappings.
        
        Returns:
            Dict: Dictionary containing feature information
        """
        return {
            "feature_order": self.feature_order,
            "categorical_mappings": self.encoder_mappings,
            "model_type": type(self.model).__name__,
            "num_features": len(self.feature_order)
        }

    def decode_prediction(self, prediction: int) -> str:
        """
        Decode prediction to human-readable format.
        
        Args:
            prediction (int): Numeric prediction (0 or 1)
            
        Returns:
            str: Human-readable prediction
        """
        return "Course Completed" if prediction == 1 else "Course Not Completed"


def main():
    """
    Main function to test the inference model.
    """
    try:
        # Initialize inference model
        inference = InferenceModel(model_dir="models/")
        
        # Test with sample input
        sample_input = {
            "country": "India",
            "hours_per_week": 10.5,
            "num_logins_last_month": 15,
            "videos_watched_pct": 75.0,
            "assignments_submitted": 5,
            "discussion_posts": 3,
            "is_working_professional": 1,
            "preferred_device": "mobile"
        }
        
        print("\nTesting inference with sample input:")
        print(f"Input: {sample_input}")
        
        # Make prediction
        prediction = inference.predict(sample_input)
        probabilities = inference.predict_proba(sample_input)
        
        print(f"Prediction: {prediction} ({inference.decode_prediction(prediction)})")
        print(f"Probabilities: {probabilities}")
        
        # Test with different country
        sample_input_2 = sample_input.copy()
        sample_input_2["country"] = "USA"
        sample_input_2["hours_per_week"] = 5.0
        
        print(f"\nTesting with different input:")
        print(f"Input: {sample_input_2}")
        
        prediction_2 = inference.predict(sample_input_2)
        probabilities_2 = inference.predict_proba(sample_input_2)
        
        print(f"Prediction: {prediction_2} ({inference.decode_prediction(prediction_2)})")
        print(f"Probabilities: {probabilities_2}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()




