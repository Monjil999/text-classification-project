import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

from src.config.config import MODELS_DIR
from src.models.model_trainer import ModelTrainer
from src.preprocessing.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_predict(input_file: str, preprocessor: DataPreprocessor, model_trainer: ModelTrainer, model_type: str) -> tuple:
    """Load data, preprocess it, and make predictions"""
    # Load and preprocess data
    df = pd.read_csv(input_file)
    X = preprocessor.transform(df)
    y_true = df['success'].values
    
    # Load model and predict
    model_trainer.load_model(MODELS_DIR, model_type)
    y_pred = model_trainer.predict(X, model_type)
    y_pred_proba = model_trainer.predict_proba(X, model_type)[:, 1]
    
    return y_true, y_pred, y_pred_proba, df

def analyze_predictions(input_file: str, output_dir: str):
    """Analyze predictions from all models"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor_state = joblib.load(MODELS_DIR / "preprocessor.joblib")
    preprocessor.__dict__.update(preprocessor_state)
    
    model_trainer = ModelTrainer()
    
    # Dictionary to store predictions and probabilities
    all_predictions = {}
    all_probabilities = {}
    
    # Analyze each model
    model_types = ['random_forest', 'svm', 'mlp', 'logistic']
    for model_type in model_types:
        logger.info(f"\nAnalyzing {model_type.upper()} predictions...")
        
        # Get predictions
        y_true, y_pred, y_pred_proba, df = load_and_predict(input_file, preprocessor, model_trainer, model_type)
        all_predictions[model_type] = y_pred
        all_probabilities[model_type] = y_pred_proba
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_true, y_pred))
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / f'{model_type}_confusion_matrix.png')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_type.upper()}')
        plt.legend()
        plt.savefig(output_dir / f'{model_type}_roc_curve.png')
        plt.close()
    
    # Load and analyze ensemble model
    logger.info("\nAnalyzing ENSEMBLE predictions...")
    y_true, y_pred, y_pred_proba, df = load_and_predict(input_file, preprocessor, model_trainer, 'ensemble')
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_true, y_pred))
    
    # Create confusion matrix plot for ensemble
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - ENSEMBLE')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / 'ensemble_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve for ensemble
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ENSEMBLE')
    plt.legend()
    plt.savefig(output_dir / 'ensemble_roc_curve.png')
    plt.close()
    
    # Save all predictions to CSV
    results_df = df.copy()
    for model_type in model_types + ['ensemble']:
        results_df[f'{model_type}_prediction'] = all_predictions.get(model_type, y_pred)
        results_df[f'{model_type}_probability'] = all_probabilities.get(model_type, y_pred_proba)
    results_df.to_csv(output_dir / 'all_predictions.csv', index=False)
    
    # Analyze disagreements between models
    disagreements = results_df[
        (results_df['random_forest_prediction'] != results_df['svm_prediction']) |
        (results_df['svm_prediction'] != results_df['mlp_prediction']) |
        (results_df['mlp_prediction'] != results_df['logistic_prediction']) |
        (results_df['ensemble_prediction'] != results_df['random_forest_prediction'])
    ]
    
    logger.info(f"\nNumber of samples with model disagreements: {len(disagreements)}")
    disagreements.to_csv(output_dir / 'model_disagreements.csv', index=False)
    
    # Plot probability distribution comparison
    plt.figure(figsize=(12, 6))
    for model_type in model_types + ['ensemble']:
        probs = all_probabilities.get(model_type, y_pred_proba)
        sns.kdeplot(probs, label=model_type)
    plt.title('Probability Distribution Comparison')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / 'probability_distributions.png')
    plt.close()

if __name__ == "__main__":
    analyze_predictions(
        input_file="data/raw/example_extracted_text.csv",
        output_dir="analysis_results"
    ) 