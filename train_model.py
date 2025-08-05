import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

class TrainModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.encoder = None
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        print(f"📥 Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print("✅ Data loaded successfully.")

    def preprocess(self):
        print("🧹 Preprocessing data...")

        # Example: Drop irrelevant columns (customize this)
        drop_cols = ['num_siblings','weight_kg','height_cm','age','has_pet','continent','education_level','preferred_device','favorite_color','birth_month']  # adjust this based on your dataset
        self.df.drop(columns=[col for col in drop_cols if col in self.df.columns], inplace=True)

        # Handling missing values (basic approach)
        self.df.fillna(method='ffill', inplace=True)

        # Separate target
        self.y = self.df["completed_course"]
        self.X = self.df.drop("completed_course", axis=1)

        # Encode categorical columns
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"🔠 Encoding categorical columns: {list(categorical_cols)}")
            self.encoder = LabelEncoder()
            for col in categorical_cols:
                self.X[col] = self.encoder.fit_transform(self.X[col])

        # Scale numerical features
        print("📏 Scaling features...")
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        print("✅ Preprocessing complete.")

    def split_data(self):
        print("🔀 Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print("✅ Data split.")

    def train(self):
        print("🚀 Training XGBoost model...")
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        self.model.fit(self.X_train, self.y_train)
        print("✅ Model trained.")

    def save_model(self):
        print("💾 Saving model and scaler...")
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        print("✅ Model saved in 'models/' directory.")

    def run_all(self):
        self.load_data()
        self.preprocess()
        self.split_data()
        self.train()
        self.save_model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost model and save it.")
    parser.add_argument("--data_path", type=str, required=True, help="/home/mohammedabdalhussain/ML_1/online_course_completion.csv")

    args = parser.parse_args()
    trainer = TrainModel(data_path=args.data_path)
    trainer.run_all()