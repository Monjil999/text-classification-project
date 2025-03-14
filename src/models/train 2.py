import argparse
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing.feature_engineering import AdvancedFeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.config.config import (
    RANDOM_STATE, TEST_SIZE, MODELS_DIR, RAW_DATA_PATH,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME
)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_data(input_path):
    """Load and preprocess the input data."""
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    return df

def train_model(model_type, input_path):
    """Train and evaluate the model."""
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load and preprocess data
    df = load_data(input_path)
    X = df.drop('success', axis=1)
    y = df['success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Initialize feature engineering
    feature_engineer = AdvancedFeatureEngineer()
    
    # Fit and transform training data
    logging.info("Performing feature engineering on training data")
    X_train_engineered = feature_engineer.fit_transform(X_train, y_train)
    
    # Transform test data
    logging.info("Performing feature engineering on test data")
    X_test_engineered = feature_engineer.transform(X_test)
    
    # Initialize and train model
    logging.info(f"Training {model_type} model")
    trainer = ModelTrainer(model_type=model_type)
    
    with mlflow.start_run(run_name=f"{model_type}_training") as run:
        # Set model type tag
        mlflow.set_tag("model_type", model_type)
        
        # Log data shapes
        mlflow.log_params({
            "train_shape": X_train_engineered.shape,
            "test_shape": X_test_engineered.shape,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE
        })
        
        # Train and evaluate model
        train_metrics, test_metrics, cv_metrics, train_cm, test_cm = trainer.train_evaluate_model(
            X_train_engineered, X_test_engineered, y_train, y_test
        )
        
        # Log metrics
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        mlflow.log_metric("cv_mean", cv_metrics['mean'])
        mlflow.log_metric("cv_std", cv_metrics['std'])
        
        # Calculate and log overfitting metrics
        overfitting_metrics = {
            'accuracy_diff': train_metrics['accuracy'] - test_metrics['accuracy'],
            'f1_diff': train_metrics['f1'] - test_metrics['f1'],
            'roc_auc_diff': train_metrics['roc_auc'] - test_metrics['roc_auc']
        }
        for name, value in overfitting_metrics.items():
            mlflow.log_metric(name, value)
        
        # Save confusion matrix plots
        train_cm_path = Path(MODELS_DIR) / f"{model_type}_train_confusion_matrix.png"
        test_cm_path = Path(MODELS_DIR) / f"{model_type}_test_confusion_matrix.png"
        trainer.plot_confusion_matrix(train_cm, f"Train Confusion Matrix - {model_type}", train_cm_path)
        trainer.plot_confusion_matrix(test_cm, f"Test Confusion Matrix - {model_type}", test_cm_path)
        
        # Log confusion matrix plots
        mlflow.log_artifact(str(train_cm_path))
        mlflow.log_artifact(str(test_cm_path))
        
        # Save and log feature importance plot if available
        if trainer.feature_importance is not None:
            importance_path = Path(MODELS_DIR) / f"{model_type}_feature_importance.png"
            trainer.plot_feature_importance(importance_path)
            mlflow.log_artifact(str(importance_path))
        
        # Save model
        model_path = Path(MODELS_DIR) / f"{model_type}_model.joblib"
        trainer.save_model(model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(trainer.model, f"{model_type}_model")
        
        # Log run ID
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")
        
        return train_metrics, test_metrics, cv_metrics, run_id

def main():
    """Main function to run the training process."""
    parser = argparse.ArgumentParser(description='Train a model for text classification')
    parser.add_argument('--model', type=str, default='logistic',
                      choices=['logistic', 'random_forest', 'svm', 'mlp', 'catboost'],
                      help='Type of model to train')
    parser.add_argument('--input', type=str, default=str(RAW_DATA_PATH),
                      help='Path to input data CSV file')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    try:
        # Train model
        train_metrics, test_metrics, cv_metrics, run_id = train_model(args.model, args.input)
        
        # Log results
        logging.info("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            logging.info(f"{metric}: {value:.3f}")
        
        logging.info("\nTest Metrics:")
        for metric, value in test_metrics.items():
            logging.info(f"{metric}: {value:.3f}")
        
        logging.info("\nCross-validation Metrics:")
        logging.info(f"Mean: {cv_metrics['mean']:.3f}")
        logging.info(f"Std: {cv_metrics['std']:.3f}")
        
        logging.info(f"\nModel saved with MLflow Run ID: {run_id}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 