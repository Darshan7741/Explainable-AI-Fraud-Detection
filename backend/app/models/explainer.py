"""SHAP and LIME explainability for fraud detection model."""
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Wrapper class for model explainability using SHAP and LIME."""
    
    def __init__(self, model_path: Optional[str] = None, data_path: Optional[str] = None):
        """
        Initialize the explainer.
        
        Args:
            model_path: Path to the saved model file
            data_path: Path to the training data CSV file
        """
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = project_root / "backend" / "trained_models" / "best_model.pkl"
        
        if data_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            data_path = project_root / "dataset.csv"
        
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.training_data = None
        self.feature_names = None
        
        self._load_model()
        self._load_training_data()
        self._initialize_explainers()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get("model")
                    self.feature_names = model_data.get("feature_names")
                else:
                    self.model = model_data
                    self.feature_names = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
                logger.info("Model loaded for explainability")
            else:
                logger.warning("Model file not found. Explainability may not work.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def _load_training_data(self):
        """Load training data for explainers."""
        try:
            if self.data_path.exists():
                df = pd.read_csv(self.data_path)
                # Remove Class column if present
                if "Class" in df.columns:
                    df = df.drop("Class", axis=1)
                # Sample for performance
                if len(df) > 1000:
                    df = df.sample(n=1000, random_state=42)
                self.training_data = df.values
                logger.info("Training data loaded for explainability")
            else:
                logger.warning("Training data not found. Using default sample.")
                # Create dummy data
                self.training_data = np.random.randn(100, 30)
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            self.training_data = np.random.randn(100, 30)
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers."""
        if self.model is None or self.training_data is None:
            return
        
        try:
            # Initialize SHAP explainer
            # Use TreeExplainer for tree-based models, KernelExplainer as fallback
            if hasattr(self.model, 'predict_proba'):
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                except:
                    # Fallback to KernelExplainer
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        self.training_data[:100]  # Use subset for performance
                    )
                logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {str(e)}")
        
        try:
            # Initialize LIME explainer
            if self.feature_names is None:
                self.feature_names = [f"Feature_{i}" for i in range(self.training_data.shape[1])]
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data,
                feature_names=self.feature_names,
                class_names=["Normal", "Fraud"],
                mode="classification"
            )
            logger.info("LIME explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize LIME explainer: {str(e)}")
    
    def explain_shap(self, features: Dict[str, float], max_features: int = 10) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            features: Dictionary with feature names and values
            max_features: Maximum number of features to return
            
        Returns:
            Dictionary with SHAP explanation
        """
        if self.shap_explainer is None or self.model is None:
            return {"error": "SHAP explainer not initialized"}
        
        try:
            # Convert features to array
            feature_array = []
            if self.feature_names:
                for feature_name in self.feature_names:
                    feature_array.append(features.get(feature_name, 0.0))
            else:
                feature_array = list(features.values())
            
            feature_array = np.array(feature_array).reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(feature_array)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use fraud class
            
            # Ensure we have a 1D array
            shap_values = np.array(shap_values).flatten()
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Get first (and only) sample
            
            # Get feature importance
            feature_importance = []
            for i, (feature_name, value) in enumerate(zip(self.feature_names, shap_values)):
                # Ensure value is a scalar
                if isinstance(value, np.ndarray):
                    value = value.item() if value.size == 1 else float(value[0])
                else:
                    value = float(value)
                
                feature_importance.append({
                    "feature": feature_name,
                    "shap_value": value,
                    "importance": abs(value)
                })
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            
            # Handle expected_value (base value)
            base_value = self.shap_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                if len(base_value) > 1:
                    base_value = base_value[1]
                else:
                    base_value = base_value[0]
            if isinstance(base_value, np.ndarray):
                base_value = base_value.item() if base_value.size == 1 else float(base_value[0])
            else:
                base_value = float(base_value)
            
            # Get prediction probability
            pred_proba = self.model.predict_proba(feature_array)[0]
            if isinstance(pred_proba, np.ndarray):
                fraud_prob = float(pred_proba[1]) if len(pred_proba) > 1 else float(pred_proba[0])
            else:
                fraud_prob = float(pred_proba)
            
            return {
                "shap_values": feature_importance[:max_features],
                "base_value": base_value,
                "prediction": fraud_prob
            }
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            return {"error": str(e)}
    
    def explain_lime(self, features: Dict[str, float], num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for a prediction.
        
        Args:
            features: Dictionary with feature names and values
            num_features: Number of features to return
            
        Returns:
            Dictionary with LIME explanation
        """
        if self.lime_explainer is None or self.model is None:
            return {"error": "LIME explainer not initialized"}
        
        try:
            # Convert features to array
            feature_array = []
            if self.feature_names:
                for feature_name in self.feature_names:
                    feature_array.append(features.get(feature_name, 0.0))
            else:
                feature_array = list(features.values())
            
            feature_array = np.array(feature_array)
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                feature_array,
                self.model.predict_proba,
                num_features=num_features
            )
            
            # Extract explanation data
            explanation_list = explanation.as_list()
            feature_importance = []
            for feature, importance in explanation_list:
                feature_importance.append({
                    "feature": feature,
                    "importance": float(importance)
                })
            
            return {
                "lime_explanation": feature_importance,
                "prediction": float(self.model.predict_proba(feature_array.reshape(1, -1))[0][1]),
                "local_prediction": float(explanation.local_pred[1] if len(explanation.local_pred) > 1 else explanation.local_pred[0])
            }
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            return {"error": str(e)}

