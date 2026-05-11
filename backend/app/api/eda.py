"""EDA API endpoint."""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from ..utils.data_loader import load_dataset, get_eda_statistics, get_feature_distributions, detect_outliers

router = APIRouter(prefix="/api/eda", tags=["eda"])

# Cache dataset (lazy loading)
_dataset = None

def get_dataset():
    """Get or load the dataset."""
    global _dataset
    if _dataset is None:
        _dataset = load_dataset()
    return _dataset


@router.get("/statistics", response_model=Dict[str, Any])
async def get_statistics():
    """
    Get comprehensive EDA statistics.
    
    Returns:
        Dictionary with dataset statistics
    """
    try:
        df = get_dataset()
        stats = get_eda_statistics(df)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading statistics: {str(e)}")


@router.get("/distributions", response_model=Dict[str, Any])
async def get_distributions(
    sample_size: int = Query(10000, ge=1000, le=50000, description="Sample size for distribution calculation")
):
    """
    Get feature distributions.
    
    Args:
        sample_size: Number of samples to use
        
    Returns:
        Dictionary with distribution data for each feature
    """
    try:
        df = get_dataset()
        distributions = get_feature_distributions(df, sample_size=sample_size)
        return distributions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading distributions: {str(e)}")


@router.get("/outliers", response_model=Dict[str, Any])
async def get_outliers(
    method: str = Query("iqr", regex="^(iqr|zscore)$", description="Outlier detection method")
):
    """
    Get outlier information.
    
    Args:
        method: Method to use ('iqr' or 'zscore')
        
    Returns:
        Dictionary with outlier information
    """
    try:
        df = get_dataset()
        outliers = detect_outliers(df, method=method)
        return outliers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting outliers: {str(e)}")


@router.get("/summary", response_model=Dict[str, Any])
async def get_summary():
    """
    Get complete EDA summary (statistics, distributions, outliers).
    
    Returns:
        Complete EDA summary
    """
    try:
        df = get_dataset()
        
        stats = get_eda_statistics(df)
        distributions = get_feature_distributions(df, sample_size=5000)
        outliers = detect_outliers(df, method="iqr")
        
        return {
            "statistics": stats,
            "distributions": distributions,
            "outliers": outliers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

