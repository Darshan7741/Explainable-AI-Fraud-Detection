"""Fraud detection model loading and prediction."""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Wrapper class for fraud detection model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the fraud detection model.
        
        Args:
            model_path: Path to the saved model file. If None, uses default path.
        """
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = project_root / "backend" / "trained_models" / "best_model.pkl"
        
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk."""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get("model")
                    self.feature_names = model_data.get("feature_names")
                else:
                    self.model = model_data
                    # Default feature names for credit card fraud dataset
                    self.feature_names = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Please train the model first.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict fraud probability for given features.
        
        Args:
            features: Dictionary with feature names and values
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Convert features to array in correct order
        feature_array = []
        if self.feature_names:
            for feature_name in self.feature_names:
                feature_array.append(features.get(feature_name, 0.0))
        else:
            # Fallback: use all provided features
            feature_array = list(features.values())
        
        feature_array = np.array(feature_array).reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]
        
        return {
            "prediction": int(prediction),
            "probability": {
                "normal": float(probabilities[0]),
                "fraud": float(probabilities[1])
            },
            "is_fraud": bool(prediction == 1),
            "confidence": float(max(probabilities))
        }
    
    def predict_batch(self, features_list: list) -> list:
        """
        Predict fraud probability for multiple transactions.
        
        Args:
            features_list: List of dictionaries with feature names and values
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        results = []
        for features in features_list:
            results.append(self.predict(features))
        
        return results

