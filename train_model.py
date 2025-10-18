"""
Machine Learning Model Training Script

This script handles the complete training pipeline for the online course completion prediction model.
It includes proper preprocessing, categorical encoding with mapping preservation, and artifact saving.

Key Features:
- Handles categorical encoding with mapping preservation
- Saves encoder mappings for consistent inference
- Uses XGBoost for classification
- Saves all required artifacts (model, scaler, encoders, mappings)
"""

import pandas as pd
import pickle
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier


class TrainModel:
    """
    Training class for the online course completion prediction model.
    
    This class handles:
    - Data loading and preprocessing
    - Categorical encoding with mapping preservation
    - Feature scaling
    - Model training with XGBoost
    - Artifact saving for inference
    """
    
    def __init__(self, data_path: str, model_dir: str = "models/"):
        """
        Initialize the training pipeline.
        
        Args:
            data_path (str): Path to the training data CSV file
            model_dir (str): Directory to save model artifacts
        """
        self.data_path = data_path
        self.model_dir = model_dir
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        
        # Dictionary to store encoders for each categorical column
        self.encoders = {}
        
        # Dictionary to store encoder mappings (for decoding)
        self.encoder_mappings = {}
        
        # Define feature columns based on the notebook analysis
        # These are the features that remain after feature selection
        self.features = [
            'country', 'hours_per_week', 'num_logins_last_month',
            'videos_watched_pct', 'assignments_submitted',
            'discussion_posts', 'is_working_professional', 'preferred_device'
        ]
        
        # Target variable
        self.target = 'completed_course'
        
        # Store feature order for consistent inference
        self.feature_order = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the training data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and preparing features.
        
        This method replicates the preprocessing steps from the notebook:
        1. Drop unnecessary columns
        2. Handle missing values
        3. Select relevant features
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Drop columns that were removed in the notebook analysis
        columns_to_drop = [
            'num_siblings', 'weight_kg', 'height_cm', 'age', 'has_pet',
            'continent', 'education_level', 'favorite_color', 'birth_month'
        ]
        
        # Drop columns that exist in the dataset
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if existing_columns_to_drop:
            df_processed = df_processed.drop(existing_columns_to_drop, axis=1)
            print(f"Dropped columns: {existing_columns_to_drop}")
        
        # Handle missing values in numerical columns
        numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        df_processed[numerical_cols] = df_processed[numerical_cols].fillna(df_processed[numerical_cols].mean())
        
        # Select only the features we need for training
        available_features = [col for col in self.features if col in df_processed.columns]
        df_processed = df_processed[available_features + [self.target]]
        
        print(f"Selected features: {available_features}")
        print(f"Final dataset shape: {df_processed.shape}")
        
        return df_processed

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder and save mappings.
        
        This method:
        1. Identifies categorical columns
        2. Creates LabelEncoder for each categorical column
        3. Saves the mapping between original values and encoded values
        4. Transforms the data
        
        Args:
            df (pd.DataFrame): Dataset with categorical features
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        print("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Identify categorical columns (object type)
        categorical_cols = df_encoded.select_dtypes(include='object').columns.tolist()
        
        print(f"Categorical columns found: {categorical_cols}")
        
        # Encode each categorical column
        for col in categorical_cols:
            if col in df_encoded.columns:
                print(f"Encoding column: {col}")
                
                # Create encoder for this column
                encoder = LabelEncoder()
                
                # Fit and transform the column
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                
                # Store the encoder
                self.encoders[col] = encoder
                
                # Create and store the mapping (original_value -> encoded_value)
                unique_values = df[col].astype(str).unique()
                mapping = {value: int(encoder.transform([value])[0]) for value in unique_values}
                self.encoder_mappings[col] = mapping
                
                print(f"  - Unique values: {len(unique_values)}")
                print(f"  - Sample mapping: {dict(list(mapping.items())[:3])}")
        
        # Convert all columns to appropriate numeric types
        df_encoded = df_encoded.astype(float)
        
        print("Categorical encoding completed.")
        return df_encoded

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Dataset with encoded features
            
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        print("Scaling numerical features...")
        
        df_scaled = df.copy()
        
        # Scale all features except target
        feature_cols = [col for col in df_scaled.columns if col != self.target]
        df_scaled[feature_cols] = self.scaler.fit_transform(df_scaled[feature_cols])
        
        # Store feature order for consistent inference
        self.feature_order = feature_cols
        
        print(f"Scaled {len(feature_cols)} features.")
        return df_scaled

    def train_model(self, df: pd.DataFrame) -> None:
        """
        Train the XGBoost model on the preprocessed data.
        
        Args:
            df (pd.DataFrame): Preprocessed and scaled dataset
        """
        print("Training XGBoost model...")
        
        # Prepare features and target
        X = df[self.feature_order]
        y = df[self.target]
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Initialize and train XGBoost model
        self.model = XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def save_artifacts(self) -> None:
        """
        Save all model artifacts required for inference.
        
        Saves:
        - Trained model (best_model.pkl)
        - Feature scaler (scaler.pkl)
        - Categorical encoders (encoders.pkl)
        - Encoder mappings (label_mappings.json)
        - Feature order (feature_order.json)
        """
        print("Saving model artifacts...")
        
        # Save the trained model
        model_path = os.path.join(self.model_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to: {model_path}")
        
        # Save the scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save the encoders
        encoders_path = os.path.join(self.model_dir, "encoders.pkl")
        with open(encoders_path, "wb") as f:
            pickle.dump(self.encoders, f)
        print(f"Encoders saved to: {encoders_path}")
        
        # Save encoder mappings as JSON for human readability
        mappings_path = os.path.join(self.model_dir, "label_mappings.json")
        with open(mappings_path, "w") as f:
            json.dump(self.encoder_mappings, f, indent=2)
        print(f"Encoder mappings saved to: {mappings_path}")
        
        # Save feature order
        feature_order_path = os.path.join(self.model_dir, "feature_order.json")
        with open(feature_order_path, "w") as f:
            json.dump(self.feature_order, f, indent=2)
        print(f"Feature order saved to: {feature_order_path}")
        
        print("All artifacts saved successfully!")

    def train(self) -> None:
        """
        Execute the complete training pipeline.
        
        This method orchestrates the entire training process:
        1. Load data
        2. Preprocess data
        3. Encode categorical features
        4. Scale features
        5. Train model
        6. Save artifacts
        """
        print("Starting model training pipeline...")
        
        # Load and preprocess data
        df = self.load_data()
        df = self.preprocess_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale features
        df = self.scale_features(df)
        
        # Train the model
        self.train_model(df)
        
        # Save all artifacts
        self.save_artifacts()
        
        print("Training pipeline completed successfully!")


def main():
    """
    Main function to run the training pipeline.
    """
    # Initialize trainer with data path
    trainer = TrainModel(
        data_path="/home/mohammedabdalhussain/ML_1/data/online_course_completion.csv",
        model_dir="models/"
    )
    
    # Run the training pipeline
    trainer.train()


if __name__ == "__main__":
    main()