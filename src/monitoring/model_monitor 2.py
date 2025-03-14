import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import logging
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, metrics_path: Optional[Path] = None):
        self.metrics_path = metrics_path or Path("monitoring/metrics")
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.performance_history: List[Dict] = []
        self.drift_history: List[Dict] = []
        
    def track_performance(self, predictions: np.ndarray, actual: np.ndarray) -> Dict:
        """Track model performance metrics over time."""
        try:
            accuracy = accuracy_score(actual, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predictions, average='weighted'
            )
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "sample_size": len(predictions)
            }
            
            self.performance_history.append(metrics)
            self._save_metrics("performance")
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    "monitoring_accuracy": accuracy,
                    "monitoring_precision": precision,
                    "monitoring_recall": recall,
                    "monitoring_f1": f1
                })
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")
            raise
    
    def check_data_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Check for data drift between reference and current data."""
        try:
            # Convert numpy arrays to DataFrames
            reference_df = pd.DataFrame(reference_data)
            current_df = pd.DataFrame(current_data)
            
            drift_metrics = {
                'ks_test_results': {},
                'mean_differences': {},
                'std_differences': {}
            }
            
            # Perform KS test for each feature
            for col in reference_df.columns:
                statistic, p_value = ks_2samp(reference_df[col], current_df[col])
                drift_metrics['ks_test_results'][f'feature_{col}'] = {
                    'statistic': statistic,
                    'p_value': p_value
                }
                
                # Calculate mean and std differences
                drift_metrics['mean_differences'][f'feature_{col}'] = abs(
                    reference_df[col].mean() - current_df[col].mean()
                )
                drift_metrics['std_differences'][f'feature_{col}'] = abs(
                    reference_df[col].std() - current_df[col].std()
                )
            
            self.drift_metrics = drift_metrics
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error checking data drift: {str(e)}")
            raise
    
    def _calculate_distribution_difference(self, dist1: pd.Series, dist2: pd.Series) -> float:
        """Calculate difference between two distributions."""
        # Align distributions on same categories
        all_categories = set(dist1.index) | set(dist2.index)
        dist1 = dist1.reindex(all_categories, fill_value=0)
        dist2 = dist2.reindex(all_categories, fill_value=0)
        
        # Calculate absolute differences
        return float(np.abs(dist1 - dist2).mean())
    
    def _save_metrics(self, metric_type: str):
        """Save metrics to disk."""
        try:
            metrics = self.performance_history if metric_type == "performance" else self.drift_history
            output_path = self.metrics_path / f"{metric_type}_metrics.json"
            
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving {metric_type} metrics: {str(e)}")
    
    def generate_monitoring_report(self, output_path: Optional[Path] = None) -> Dict:
        """Generate a comprehensive monitoring report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "performance_summary": self._generate_performance_summary(),
                "drift_summary": self._generate_drift_summary(),
                "alerts": self._generate_alerts()
            }
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=4)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {str(e)}")
            raise
    
    def _generate_performance_summary(self) -> Dict:
        """Generate summary of performance metrics."""
        if not self.performance_history:
            return {}
            
        metrics_df = pd.DataFrame(self.performance_history)
        return {
            "latest_metrics": metrics_df.iloc[-1].to_dict(),
            "metric_trends": {
                col: {
                    "mean": metrics_df[col].mean(),
                    "std": metrics_df[col].std(),
                    "min": metrics_df[col].min(),
                    "max": metrics_df[col].max()
                }
                for col in ["accuracy", "precision", "recall", "f1"]
            }
        }
    
    def _generate_drift_summary(self) -> Dict:
        """Generate summary of drift metrics."""
        if not self.drift_history:
            return {}
            
        latest_drift = self.drift_history[-1]
        drifted_features = [
            feature for feature, metrics in latest_drift["feature_drifts"].items()
            if metrics.get("is_drift", False)
        ]
        
        return {
            "drifted_features": drifted_features,
            "total_features_checked": len(latest_drift["feature_drifts"]),
            "drift_detected": bool(drifted_features)
        }
    
    def _generate_alerts(self) -> List[Dict]:
        """Generate alerts based on monitoring metrics."""
        alerts = []
        
        # Performance degradation alerts
        if self.performance_history:
            latest_metrics = self.performance_history[-1]
            if latest_metrics["accuracy"] < 0.8:  # Threshold can be configured
                alerts.append({
                    "type": "performance",
                    "severity": "high",
                    "message": f"Low accuracy detected: {latest_metrics['accuracy']:.3f}"
                })
        
        # Drift alerts
        if self.drift_history:
            latest_drift = self.drift_history[-1]
            drifted_features = [
                feature for feature, metrics in latest_drift["feature_drifts"].items()
                if metrics.get("is_drift", False)
            ]
            if drifted_features:
                alerts.append({
                    "type": "drift",
                    "severity": "medium",
                    "message": f"Data drift detected in features: {', '.join(drifted_features)}"
                })
        
        return alerts 