"""Explainability API endpoint."""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal
from ..models.explainer import ModelExplainer

router = APIRouter(prefix="/api/explain", tags=["explainability"])

# Initialize explainer (lazy loading)
_explainer = None

def get_explainer():
    """Get or initialize the model explainer."""
    global _explainer
    if _explainer is None:
        _explainer = ModelExplainer()
    return _explainer


class ExplainRequest(BaseModel):
    """Request model for explanation."""
    Time: float = Field(..., description="Transaction time")
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")
    Amount: float = Field(..., description="Transaction amount")


@router.post("/shap", response_model=Dict[str, Any])
async def explain_shap(
    request: ExplainRequest,
    max_features: int = Query(10, ge=1, le=30, description="Maximum number of features to return")
):
    """
    Generate SHAP explanation for a prediction.
    
    Args:
        request: Transaction features
        max_features: Maximum number of features to return
        
    Returns:
        SHAP explanation with feature importance
    """
    try:
        explainer = get_explainer()
        features = request.dict()
        result = explainer.explain_shap(features, max_features=max_features)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@router.post("/lime", response_model=Dict[str, Any])
async def explain_lime(
    request: ExplainRequest,
    num_features: int = Query(10, ge=1, le=30, description="Number of features to return")
):
    """
    Generate LIME explanation for a prediction.
    
    Args:
        request: Transaction features
        num_features: Number of features to return
        
    Returns:
        LIME explanation with feature importance
    """
    try:
        explainer = get_explainer()
        features = request.dict()
        result = explainer.explain_lime(features, num_features=num_features)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@router.post("/both", response_model=Dict[str, Any])
async def explain_both(
    request: ExplainRequest,
    max_features: int = Query(10, ge=1, le=30, description="Maximum number of features to return")
):
    """
    Generate both SHAP and LIME explanations for a prediction.
    
    Args:
        request: Transaction features
        max_features: Maximum number of features to return
        
    Returns:
        Both SHAP and LIME explanations
    """
    try:
        explainer = get_explainer()
        features = request.dict()
        
        shap_result = explainer.explain_shap(features, max_features=max_features)
        lime_result = explainer.explain_lime(features, num_features=max_features)
        
        return {
            "shap": shap_result,
            "lime": lime_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

