import argparse
import logging
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.config.config import MODELS_DIR
from src.models.model_trainer import ModelTrainer
from src.monitoring.model_monitor import ModelMonitor
from src.utils.data_validation import DataValidator, validate_data
from src.utils.feature_engineering import AdvancedFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    try:
        # Load and preprocess input data
        logger.info("Loading input data...")
        df = pd.read_csv(args.input)
        
        # Extract features
        logger.info("Extracting features...")
        feature_engineer = AdvancedFeatureEngineer()
        features = feature_engineer.create_features(df)
        
        # Initialize model trainer and load model
        logger.info(f"Loading {args.model} model...")
        model_trainer = ModelTrainer(model_type=args.model)
        model_trainer.load(
            model_path=os.path.join(MODELS_DIR, f"{args.model}_model.joblib"),
            preprocessor_path=os.path.join(MODELS_DIR, f"{args.model}_preprocessor.joblib")
        )
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model_trainer.predict(features)
        probabilities = model_trainer.predict_proba(features)
        prediction_confidence = np.max(probabilities, axis=1)
        
        # Create output DataFrame
        logger.info("Saving predictions...")
        output_df = df.copy()
        output_df['predicted_label'] = predictions
        output_df['prediction_confidence'] = prediction_confidence
        
        # Save predictions
        output_df.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")
        
        # Monitor predictions
        logger.info("Running monitoring checks...")
        monitor = ModelMonitor()
        monitor.check_data_drift(features)
        monitor.track_performance(predictions, prediction_confidence)
        monitor.save_metrics()
        
        logger.info("Prediction process completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument("--input", required=True, help="Path to input data CSV file")
    parser.add_argument("--model", required=True, choices=['logistic', 'random_forest', 'svm', 'mlp', 'catboost'], help="Model type to use")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    
    args = parser.parse_args()
    main(args)