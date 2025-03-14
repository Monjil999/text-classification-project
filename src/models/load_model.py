import mlflow
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_registered_model():
    """Load the registered model from MLflow."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri('mlruns')
        
        # Load the registered model
        logger.info("Loading registered model 'text_classification_logistic'...")
        model = mlflow.pyfunc.load_model("models:/text_classification_logistic/1")
        logger.info("Model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

if __name__ == "__main__":
    load_registered_model() 