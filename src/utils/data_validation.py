import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TextDataSchema(BaseModel):
    text: str = Field(..., min_length=1)
    label: int = Field(..., ge=0)

def validate_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate input data for predictions.
    
    Args:
        data: DataFrame containing input data
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Check for null values
        null_counts = data.isnull().sum()
        if null_counts.any():
            validation_results['warnings'].append(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=np.number)).sum()
        if inf_counts.any():
            validation_results['errors'].append(f"Found infinite values in columns: {inf_counts[inf_counts > 0].to_dict()}")
            validation_results['is_valid'] = False
        
        # Check data types
        numeric_cols = data.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            validation_results['warnings'].append("No numeric columns found in the data")
        
        return validation_results
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Validation error: {str(e)}")
        return validation_results

class DataValidator:
    def __init__(self, schema: BaseModel = TextDataSchema):
        self.schema = schema
        self.quality_metrics: Dict = {}
        self.logger = logging.getLogger(__name__)
        
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """Validate data against the defined schema."""
        try:
            for _, row in data.iterrows():
                self.schema(**row.to_dict())
            return True
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return False

    def check_data_quality(self, data: pd.DataFrame) -> Dict:
        """Perform data quality checks."""
        self.quality_metrics = {
            "missing_values": data.isnull().sum().to_dict(),
            "duplicates": data.duplicated().sum(),
            "empty_text": (data["text"].str.strip() == "").sum(),
            "label_distribution": data["label"].value_counts().to_dict(),
            "text_length_stats": {
                "mean": data["text"].str.len().mean(),
                "min": data["text"].str.len().min(),
                "max": data["text"].str.len().max()
            }
        }
        return self.quality_metrics

    def validate_output(self, predictions: np.ndarray, expected_classes: List[int]) -> bool:
        """Validate model output predictions."""
        try:
            unique_classes = np.unique(predictions)
            if not all(cls in expected_classes for cls in unique_classes):
                logger.error(f"Invalid prediction classes found: {unique_classes}")
                return False
            return True
        except Exception as e:
            logger.error(f"Output validation failed: {str(e)}")
            return False

    def generate_validation_report(self, data: pd.DataFrame, output_path: Optional[Path] = None) -> Dict:
        """Generate a comprehensive validation report."""
        report = {
            "schema_validation": self.validate_schema(data),
            "quality_metrics": self.check_data_quality(data),
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                import json
                json.dump(report, f, indent=4, default=str)
        
        return report

    def validate_features(self, features: np.ndarray, feature_names: List[str]) -> bool:
        """Validate engineered features."""
        try:
            if features.shape[1] != len(feature_names):
                logger.error(f"Feature dimension mismatch: {features.shape[1]} != {len(feature_names)}")
                return False
            
            if np.isnan(features).any():
                logger.error("NaN values found in features")
                return False
                
            if np.isinf(features).any():
                logger.error("Infinite values found in features")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Feature validation failed: {str(e)}")
            return False

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a validation report for the input data."""
        return validate_data(data) 