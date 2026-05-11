"""CLI script for training fraud detection models."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Lazy imports for optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def load_data(data_path: str):
    """Load the dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    return df


def preprocess_data(df: pd.DataFrame, use_smote: bool = True, test_size: float = 0.2):
    """Preprocess the data."""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    if use_smote:
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE - Class distribution:\n{pd.Series(y_train).value_counts()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None):
    """Train Random Forest model."""
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nRandom Forest Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model, roc_auc_score(y_test, y_pred_proba)


def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6):
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Please install it or use another model.")
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc',
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model, roc_auc_score(y_test, y_pred_proba)


def train_lightgbm(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6):
    """Train LightGBM model."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not available. Please install it or use another model.")
    print("\nTraining LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nLightGBM Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model, roc_auc_score(y_test, y_pred_proba)


def save_model(model, scaler, feature_names, model_path: str, model_name: str):
    """Save the trained model."""
    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }
    joblib.dump(model_data, model_path)
    print(f"\n{model_name} saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--data",
        type=str,
        default="../dataset.csv",
        help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../backend/trained_models/best_model.pkl",
        help="Path to save the best model"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "xgb", "lgb", "all"],
        default="all",
        help="Model to train (rf=RandomForest, xgb=XGBoost, lgb=LightGBM, all=all models)"
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE for class balancing"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators for tree-based models"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for tree-based models"
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
        df, use_smote=not args.no_smote
    )
    
    # Train models
    models = {}
    scores = {}
    
    if args.model in ["rf", "all"]:
        model, score = train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        models["rf"] = model
        scores["rf"] = score
    
    if args.model in ["xgb", "all"]:
        model, score = train_xgboost(
            X_train, y_train, X_test, y_test,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or 6
        )
        models["xgb"] = model
        scores["xgb"] = score
    
    if args.model in ["lgb", "all"]:
        model, score = train_lightgbm(
            X_train, y_train, X_test, y_test,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or 6
        )
        models["lgb"] = model
        scores["lgb"] = score
    
    # Select best model
    if len(scores) > 0:
        best_model_name = max(scores, key=scores.get)
        best_model = models[best_model_name]
        best_score = scores[best_model_name]
        
        print(f"\n{'='*50}")
        print(f"Best Model: {best_model_name.upper()} (ROC-AUC: {best_score:.4f})")
        print(f"{'='*50}")
        
        # Save best model
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(best_model, scaler, feature_names, str(output_path), best_model_name.upper())
        
        print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

