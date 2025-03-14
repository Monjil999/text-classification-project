import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict
import joblib
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.scaler = StandardScaler()
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform the training data
        """
        # Separate features and target
        X = df.iloc[:, :-1]
        y = df['success']
        
        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Encode categorical variables
        X_encoded = X_imputed.copy()
        for column in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X_encoded[column])
            self.label_encoders[column] = le
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_encoded)
        
        return X_scaled, y.values
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform test/new data using fitted preprocessor
        """
        # Handle missing values
        X = df.iloc[:, :-1] if 'success' in df.columns else df
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        
        # Encode categorical variables
        X_encoded = X_imputed.copy()
        for column in X_encoded.columns:
            if column in self.label_encoders:
                le = self.label_encoders[column]
                X_encoded[column] = le.transform(X_encoded[column])
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        return X_scaled
    
    def save(self, path: Path):
        """Save preprocessor state"""
        state = {
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'scaler': self.scaler
        }
        joblib.dump(state, path / 'preprocessor.joblib')
    
    def load(self, path: Path):
        """Load preprocessor state"""
        state = joblib.load(path / 'preprocessor.joblib')
        self.label_encoders = state['label_encoders']
        self.imputer = state['imputer']
        self.scaler = state['scaler'] 