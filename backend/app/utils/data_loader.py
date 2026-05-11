"""Data loading utilities for the fraud detection system."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Load the fraud detection dataset.
    
    Args:
        file_path: Path to the dataset CSV file. If None, uses default path.
        
    Returns:
        DataFrame containing the dataset
    """
    if file_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        file_path = project_root / "dataset.csv"
    
    df = pd.read_csv(file_path)
    return df


def get_eda_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive EDA statistics for the dataset.
    
    Args:
        df: DataFrame containing the dataset
        
    Returns:
        Dictionary with EDA statistics
    """
    stats = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "class_distribution": {
            "normal": int((df["Class"] == 0).sum()),
            "fraud": int((df["Class"] == 1).sum()),
            "fraud_percentage": float((df["Class"] == 1).sum() / len(df) * 100)
        },
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict(),
        "feature_names": df.columns.tolist(),
        "correlation_matrix": df.corr().to_dict()
    }
    
    return stats


def get_feature_distributions(df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
    """
    Get distribution data for features (sampled for performance).
    
    Args:
        df: DataFrame containing the dataset
        sample_size: Number of samples to use for distribution calculation
        
    Returns:
        Dictionary with distribution data for each feature
    """
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    distributions = {}
    for col in df.columns:
        if col != "Class":
            distributions[col] = {
                "values": df_sample[col].tolist(),
                "min": float(df_sample[col].min()),
                "max": float(df_sample[col].max()),
                "mean": float(df_sample[col].mean()),
                "std": float(df_sample[col].std()),
                "median": float(df_sample[col].median())
            }
    
    return distributions


def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> Dict[str, Any]:
    """
    Detect outliers in the dataset.
    
    Args:
        df: DataFrame containing the dataset
        method: Method to use for outlier detection ('iqr' or 'zscore')
        
    Returns:
        Dictionary with outlier information
    """
    outliers = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Class" in numeric_cols:
        numeric_cols.remove("Class")
    
    for col in numeric_cols[:10]:  # Limit to first 10 features for performance
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        else:  # zscore
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_count = (z_scores > 3).sum()
        
        outliers[col] = {
            "count": int(outlier_count),
            "percentage": float(outlier_count / len(df) * 100)
        }
    
    return outliers

