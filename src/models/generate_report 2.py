import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tabulate import tabulate
from src.config.config import MODEL_CONFIGS, MLFLOW_TRACKING_URI, EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_runs():
    """Get the latest run for each model type from MLflow."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        raise ValueError(f"No experiment found with name {EXPERIMENT_NAME}")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"]
    )
    
    # Group runs by model type and get the latest for each
    latest_runs = {}
    for run in runs:
        model_type = run.data.tags.get("model_type")
        if model_type and model_type not in latest_runs:
            latest_runs[model_type] = run
    
    return latest_runs

def calculate_overfitting_scores(metrics):
    """Calculate overfitting scores based on metrics."""
    return {
        'accuracy_gap': metrics.get('train_accuracy', 0) - metrics.get('test_accuracy', 0),
        'f1_gap': metrics.get('train_f1', 0) - metrics.get('test_f1', 0),
        'roc_auc_gap': metrics.get('train_roc_auc', 0) - metrics.get('test_roc_auc', 0),
        'cv_std': metrics.get('cv_std', 0)
    }

def generate_report():
    """Generate a comprehensive training report."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    latest_runs = get_latest_runs()
    
    # Prepare data for report
    report_data = []
    overfitting_warnings = []
    
    for model_type, run in latest_runs.items():
        metrics = run.data.metrics
        params = run.data.params
        
        # Calculate overfitting scores
        overfitting_scores = calculate_overfitting_scores(metrics)
        
        # Check for overfitting
        if overfitting_scores['accuracy_gap'] > 0.1:
            overfitting_warnings.append(
                f"{model_type}: Large accuracy gap ({overfitting_scores['accuracy_gap']:.3f})"
            )
        if overfitting_scores['cv_std'] > 0.1:
            overfitting_warnings.append(
                f"{model_type}: High CV variance ({overfitting_scores['cv_std']:.3f})"
            )
        
        # Collect metrics for report
        report_data.append({
            'Model': model_type,
            'Train Accuracy': metrics.get('train_accuracy', 0),
            'Test Accuracy': metrics.get('test_accuracy', 0),
            'Train F1': metrics.get('train_f1', 0),
            'Test F1': metrics.get('test_f1', 0),
            'Train ROC-AUC': metrics.get('train_roc_auc', 0),
            'Test ROC-AUC': metrics.get('test_roc_auc', 0),
            'CV Mean': metrics.get('cv_mean', 0),
            'CV Std': metrics.get('cv_std', 0),
            'Accuracy Gap': overfitting_scores['accuracy_gap'],
            'Best Parameters': str(params)
        })
    
    # Create report directory
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)
    
    # Generate report file
    report_path = report_dir / 'training_report.txt'
    with open(report_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("MODEL TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Write model comparison table
        df = pd.DataFrame(report_data)
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.3f'))
        f.write("\n\n")
        
        # Write overfitting analysis
        f.write("OVERFITTING ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if overfitting_warnings:
            f.write("\nWarnings:\n")
            for warning in overfitting_warnings:
                f.write(f"- {warning}\n")
        else:
            f.write("\nNo significant overfitting detected.\n")
        
        # Write best model recommendation
        best_model = max(report_data, key=lambda x: x['Test F1'])
        f.write("\nBEST MODEL RECOMMENDATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best performing model: {best_model['Model']}\n")
        f.write(f"Test F1 Score: {best_model['Test F1']:.3f}\n")
        f.write(f"Test Accuracy: {best_model['Test Accuracy']:.3f}\n")
        f.write(f"Test ROC-AUC: {best_model['Test ROC-AUC']:.3f}\n")
    
    logger.info(f"Report generated at {report_path}")
    return report_path

if __name__ == "__main__":
    generate_report() 