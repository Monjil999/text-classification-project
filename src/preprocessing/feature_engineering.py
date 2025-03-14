import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=100,  # Restored to 100 features
            ngram_range=(1, 2),  # Added bigrams
            analyzer='word',
            min_df=2  # Ignore very rare terms
        )
        self.scaler = StandardScaler()
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.onehot = OneHotEncoder(handle_unknown='ignore')  # Removed sparse parameter
        self.fitted = False
        self.categorical_columns = None
        self.numeric_columns = None
        
    def fit(self, X, y=None):
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Identify column types
        self.categorical_columns = []
        self.numeric_columns = []
        for col in X.columns:
            n_unique = X[col].nunique()
            if n_unique < 10 or X[col].dtype == 'object':  # Categorical if few unique values or object type
                self.categorical_columns.append(col)
            else:
                self.numeric_columns.append(col)
        
        # Prepare text features from all columns
        text_data = X.astype(str).apply(lambda x: ' '.join(x), axis=1)
        self.tfidf.fit(text_data)
        
        # Handle categorical features
        if self.categorical_columns:
            cat_data = X[self.categorical_columns].copy()
            cat_data = self.categorical_imputer.fit_transform(cat_data)
            self.onehot.fit(cat_data)
        
        # Handle numeric features
        if self.numeric_columns:
            num_data = X[self.numeric_columns].copy()
            num_data = pd.DataFrame(num_data, columns=self.numeric_columns)
            num_data = num_data.apply(pd.to_numeric, errors='coerce')
            num_data = self.numeric_imputer.fit_transform(num_data)
            self.scaler.fit(num_data)
        
        # Create and fit pattern features
        pattern_features = self._create_pattern_features(X)
        if pattern_features is not None:
            self.pattern_scaler = StandardScaler()
            self.pattern_scaler.fit(pattern_features)
        
        self.fitted = True
        return self
        
    def transform(self, X):
        if not self.fitted:
            raise ValueError("AdvancedFeatureEngineer must be fitted first")
        
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        features_list = []
        
        # Transform text features
        text_data = X.astype(str).apply(lambda x: ' '.join(x), axis=1)
        tfidf_features = self.tfidf.transform(text_data).toarray()
        features_list.append(tfidf_features)
        logger.info(f"TF-IDF features shape: {tfidf_features.shape}")
        
        # Transform categorical features
        if self.categorical_columns:
            cat_data = X[self.categorical_columns].copy()
            cat_data = self.categorical_imputer.transform(cat_data)
            cat_features = self.onehot.transform(cat_data).toarray()  # Convert sparse matrix to dense
            features_list.append(cat_features)
            logger.info(f"Categorical features shape: {cat_features.shape}")
        
        # Transform numeric features
        if self.numeric_columns:
            num_data = X[self.numeric_columns].copy()
            num_data = pd.DataFrame(num_data, columns=self.numeric_columns)
            num_data = num_data.apply(pd.to_numeric, errors='coerce')
            num_data = self.numeric_imputer.transform(num_data)
            num_features = self.scaler.transform(num_data)
            features_list.append(num_features)
            logger.info(f"Numeric features shape: {num_features.shape}")
        
        # Transform pattern features
        pattern_features = self._create_pattern_features(X)
        if pattern_features is not None:
            pattern_features = self.pattern_scaler.transform(pattern_features)
            features_list.append(pattern_features)
            logger.info(f"Pattern features shape: {pattern_features.shape}")
        
        # Combine all features
        result = np.hstack(features_list)
        logger.info(f"Final features shape: {result.shape}")
        
        # Final check for NaN values
        if np.any(np.isnan(result)):
            logger.warning("NaN values found in final features, replacing with 0")
            result = np.nan_to_num(result, 0)
        
        return result
    
    def _create_pattern_features(self, X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        
        features = []
        
        # Special patterns per column
        for col in X.columns:
            # Length of values
            lengths = X[col].astype(str).apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            features.append(lengths.values.reshape(-1, 1))
            
            # Count of special characters
            special = X[col].astype(str).apply(lambda x: len([c for c in str(x) if not c.isalnum() and not c.isspace()]) if pd.notna(x) else 0)
            features.append(special.values.reshape(-1, 1))
            
            # Count of numbers
            numbers = X[col].astype(str).apply(lambda x: sum(c.isdigit() for c in str(x)) if pd.notna(x) else 0)
            features.append(numbers.values.reshape(-1, 1))
            
            # Count of uppercase letters
            upper = X[col].astype(str).apply(lambda x: sum(c.isupper() for c in str(x)) if pd.notna(x) else 0)
            features.append(upper.values.reshape(-1, 1))
            
            # Word count
            words = X[col].astype(str).apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
            features.append(words.values.reshape(-1, 1))
        
        return np.hstack(features) if features else None 