import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
from typing import Dict, Any, Tuple, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.preprocessing import StandardScaler

from src.config.config import MODEL_CONFIGS, RANDOM_STATE, TEST_SIZE, N_FOLDS, MODELS_DIR, ENSEMBLE_WEIGHTS
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.feature_engineering import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type: str):
        """Initialize the model trainer with specified model type."""
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        self.model = MODEL_CONFIGS[model_type]['model']
        self.param_grid = MODEL_CONFIGS[model_type]['param_grid']
        self.feature_importance = None
        self.best_model = None
    
    def train_evaluate_model(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], np.ndarray, np.ndarray]:
        """Train and evaluate the model, returning separate metrics and confusion matrices."""
        try:
            # Perform grid search
            cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                self.model,
                self.param_grid,
                cv=cv,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                refit='f1',
                n_jobs=-1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            self.best_model = grid_search.best_estimator_
            
            # Get predictions
            y_train_pred = self.best_model.predict(X_train)
            y_train_proba = self.best_model.predict_proba(X_train)[:, 1]
            
            y_test_pred = self.best_model.predict(X_test)
            y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
            
            # Calculate training metrics
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred),
                'f1': f1_score(y_train, y_train_pred),
                'roc_auc': roc_auc_score(y_train, y_train_proba)
            }
            
            # Calculate test metrics
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1': f1_score(y_test, y_test_pred),
                'roc_auc': roc_auc_score(y_test, y_test_proba)
            }
            
            # Calculate cross-validation metrics
            cv_scores = cross_val_score(
                self.best_model,
                X_train,
                y_train,
                cv=cv,
                scoring='f1'
            )
            cv_metrics = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            # Get confusion matrices
            train_cm = confusion_matrix(y_train, y_train_pred)
            test_cm = confusion_matrix(y_test, y_test_pred)
            
            # Store feature importance if available
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                self.feature_importance = self.best_model.coef_[0]
            
            return train_metrics, test_metrics, cv_metrics, train_cm, test_cm
            
        except Exception as e:
            logger.error(f"Error in train_evaluate_model: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str, save_path: Path) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
    
    def plot_feature_importance(self, save_path: Path, top_n: int = 30) -> None:
        """Plot and save feature importance."""
        if self.feature_importance is None:
            logger.warning("No feature importance available to plot")
            return
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Top {top_n} Important Features - {self.model_type.upper()}")
        importance_indices = np.argsort(np.abs(self.feature_importance))[-top_n:]
        plt.barh(range(top_n), np.abs(self.feature_importance[importance_indices]))
        plt.yticks(range(top_n), [f"Feature {i}" for i in importance_indices])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def save_model(self, path: Path) -> None:
        """Save the trained model."""
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing file if it exists
        if path.exists():
            path.unlink()
        
        # Save the model
        joblib.dump(self.best_model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load a trained model."""
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        
        self.best_model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.best_model is None:
            raise ValueError("No trained model available")
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates."""
        if self.best_model is None:
            raise ValueError("No trained model available")
        return self.best_model.predict_proba(X)

    def train_all_models(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train and evaluate all models"""
        results = {}
        
        # Train individual models
        for model_type in MODEL_CONFIGS.keys():
            print(f"\nTraining {model_type}...")
            
            with mlflow.start_run(run_name=model_type):
                metrics = self.train_evaluate_model(
                    X_train, X_test, y_train, y_test
                )
                
                mlflow.log_params(self.best_model.get_params())
                mlflow.log_metrics(metrics[0])
                mlflow.log_metrics(metrics[1])
                mlflow.log_metrics(metrics[2])
                mlflow.sklearn.log_model(self.best_model, model_type)
                
                results[model_type] = {
                    'train_metrics': metrics[0],
                    'test_metrics': metrics[1],
                    'cv_metrics': metrics[2],
                    'train_confusion_matrix': metrics[3],
                    'test_confusion_matrix': metrics[4]
                }
        
        # Create and train weighted voting ensemble
        estimators = [(name, MODEL_CONFIGS[name]['model']) for name in MODEL_CONFIGS.keys()]
        weights = [ENSEMBLE_WEIGHTS[name] for name in MODEL_CONFIGS.keys()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        # Apply feature engineering for ensemble
        X_train_engineered = self.feature_engineer.transform(X_train)
        X_test_engineered = self.feature_engineer.transform(X_test)
        
        voting_clf.fit(X_train_engineered, y_train)
        
        # Evaluate ensemble
        y_pred = voting_clf.predict(X_test_engineered)
        y_pred_proba = voting_clf.predict_proba(X_test_engineered)[:, 1]
        
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        results['ensemble'] = ensemble_metrics
        self.best_model = voting_clf
        
        # Plot ensemble confusion matrix
        self.plot_confusion_matrix(confusion_matrix(y_test, y_pred), 'ensemble', MODELS_DIR.parent / 'ensemble_confusion_matrix.png')
        
        return results
    
    def _analyze_feature_importance(self, importance_values, X, model_type):
        """Analyze and plot feature importance"""
        try:
            # Get feature names from TF-IDF
            tfidf_features = [f"tfidf_{i}" for i in range(100)]  # 100 TF-IDF features
            
            # Get categorical feature names
            if hasattr(self.feature_engineer, 'onehot') and hasattr(self.feature_engineer.onehot, 'get_feature_names_out'):
                cat_features = self.feature_engineer.onehot.get_feature_names_out(
                    self.feature_engineer.categorical_columns
                ).tolist()
            else:
                cat_features = [f"cat_{i}" for i in range(1433)]  # 1433 categorical features
            
            # Pattern feature names
            pattern_features = [
                f"{col}_{pattern}" for col in X.columns 
                for pattern in ['length', 'special', 'numbers', 'upper', 'words']
            ]
            
            # Combine all feature names
            feature_names = tfidf_features + cat_features + pattern_features
            
            # Get top features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(importance_values)  # Use absolute values for logistic regression coefficients
            }).sort_values('importance', ascending=False)
            
            # Plot top 30 features
            plt.figure(figsize=(12, 6))
            plt.title(f"Top 30 Important Features - {model_type.upper()}")
            plt.bar(range(30), importance_df['importance'][:30])
            plt.xticks(range(30), importance_df['feature'][:30], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(MODELS_DIR.parent / f'{model_type}_feature_importance.png')
            plt.close()
            
            # Save feature importance to CSV
            importance_df.to_csv(MODELS_DIR.parent / f'{model_type}_feature_importance.csv', index=False)
            
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {str(e)}")
    
    def save_model(self, path: Path):
        """Save all models and feature engineer"""
        path.mkdir(parents=True, exist_ok=True)
        
        for model_type, model in MODEL_CONFIGS.items():
            model_path = path / f"{model_type}_model.joblib"
            joblib.dump(model['model'], model_path)
        
        # Save feature engineer
        if hasattr(self, 'feature_engineer') and self.feature_engineer.fitted:
            joblib.dump(self.feature_engineer, path / "feature_engineer.joblib")
    
    def load_preprocessor(self, path: Union[str, Path]):
        """Load the preprocessor from disk."""
        try:
            from joblib import load
            path = Path(path)
            preprocessor_path = path / "preprocessor.joblib"
            feature_engineer_path = path / "feature_engineer.joblib"
            
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            if not feature_engineer_path.exists():
                raise FileNotFoundError(f"Feature engineer not found at {feature_engineer_path}")
                
            self.feature_engineer = load(feature_engineer_path)
            logger.info(f"Successfully loaded feature engineer")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise

    def preprocess_data(self, text_data: pd.Series) -> np.ndarray:
        """Preprocess text data using the loaded preprocessor."""
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not loaded. Call load_preprocessor first.")
        return self.feature_engineer.transform(text_data)
    
    def predict(self, X: np.ndarray, model_type: str = 'ensemble') -> np.ndarray:
        """Make predictions using specified model"""
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_type} not found")
            
        return MODEL_CONFIGS[model_type]['model'].predict(X)
    
    def predict_proba(self, X: np.ndarray, model_type: str = 'ensemble') -> np.ndarray:
        """Get probability estimates"""
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_type} not found")
            
        return MODEL_CONFIGS[model_type]['model'].predict_proba(X) 