# Financial Fraud Detection with Explainable AI

A comprehensive machine learning system for detecting financial fraud with advanced explainability features using SHAP and LIME. The project includes a full-stack application with a FastAPI backend, React frontend with Tailwind CSS, and comprehensive EDA capabilities.

## Features

- **Advanced EDA Dashboard**: Interactive exploratory data analysis with visualizations
  - Statistical summaries
  - Class imbalance visualization
  - Feature distributions
  - Correlation matrices
  - Outlier detection

- **Fraud Prediction**: Real-time fraud detection with confidence scores
  - Input transaction features
  - Get instant predictions
  - View probability distributions

- **Model Explainability**: Understand model decisions
  - SHAP (SHapley Additive exPlanations) visualizations
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Feature importance analysis

- **Model Training**: Flexible training options
  - Jupyter notebook for interactive exploration
  - CLI script for automated training
  - Support for multiple algorithms (Random Forest, XGBoost, LightGBM)

## Project Structure

```
/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── models/          # Model loading and explainability
│   │   └── utils/           # Utility functions
│   ├── trained_models/      # Saved model files
│   └── requirements.txt
├── frontend/                # React + TypeScript frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   └── services/        # API client
│   └── package.json
├── training/                # Model training scripts
│   ├── train_model.ipynb   # Jupyter notebook
│   ├── train_model.py       # CLI script
│   └── requirements.txt
└── dataset.csv             # Training dataset
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## Installation

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

### 3. Training Setup

```bash
cd training
pip install -r requirements.txt
```

## Usage

### Training the Model

#### Option 1: Using Jupyter Notebook

```bash
cd training
jupyter notebook train_model.ipynb
```

Run all cells in the notebook to train models and save the best one.

#### Option 2: Using CLI Script

```bash
cd training
python train_model.py --data ../dataset.csv --output ../backend/trained_models/best_model.pkl
```

**CLI Options:**
- `--data`: Path to dataset CSV file (default: `../dataset.csv`)
- `--output`: Path to save the trained model (default: `../backend/trained_models/best_model.pkl`)
- `--model`: Model to train - `rf`, `xgb`, `lgb`, or `all` (default: `all`)
- `--no-smote`: Disable SMOTE for class balancing
- `--n-estimators`: Number of estimators (default: 100)
- `--max-depth`: Maximum depth for trees (default: None)

**Example:**
```bash
# Train only XGBoost
python train_model.py --model xgb --n-estimators 200

# Train all models without SMOTE
python train_model.py --model all --no-smote
```

### Running the Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation (Swagger UI): `http://localhost:8000/docs`

### Running the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### Prediction
- `POST /api/predict/` - Predict fraud for a transaction

### Explainability
- `POST /api/explain/shap` - Get SHAP explanation
- `POST /api/explain/lime` - Get LIME explanation
- `POST /api/explain/both` - Get both SHAP and LIME explanations

### EDA
- `GET /api/eda/statistics` - Get dataset statistics
- `GET /api/eda/distributions` - Get feature distributions
- `GET /api/eda/outliers` - Get outlier information
- `GET /api/eda/summary` - Get complete EDA summary

## Dataset

The project uses a credit card fraud detection dataset with the following features:
- `Time`: Transaction time
- `V1-V28`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount
- `Class`: Target variable (0 = Normal, 1 = Fraud)

## Model Training Details

The training pipeline includes:
1. **Data Preprocessing**: Standard scaling and SMOTE for class balancing
2. **Model Training**: Multiple algorithms (Random Forest, XGBoost, LightGBM)
3. **Model Evaluation**: ROC-AUC, precision, recall, F1-score
4. **Model Selection**: Best model based on ROC-AUC score
5. **Model Saving**: Saved with scaler and feature names

## Explainability Methods

### SHAP (SHapley Additive exPlanations)
- Provides global and local explanations
- Shows feature contributions to predictions
- Uses Shapley values from game theory

### LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions
- Creates local approximations of the model
- Model-agnostic approach

## Technologies Used

### Backend
- FastAPI - Modern Python web framework
- scikit-learn - Machine learning library
- XGBoost, LightGBM - Gradient boosting frameworks
- SHAP, LIME - Explainability libraries
- pandas, numpy - Data processing

### Frontend
- React 18 - UI library
- TypeScript - Type safety
- Tailwind CSS - Utility-first CSS framework
- Chart.js - Charting library
- Plotly.js - Interactive visualizations
- React Router - Routing

## Development

### Backend Development
```bash
cd backend
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm run dev
```

## Troubleshooting

### Model Not Found Error
If you see "Model not loaded" errors, make sure to train the model first:
```bash
cd training
python train_model.py
```

### Port Already in Use
If port 8000 or 3000 is already in use:
- Backend: Change port in `uvicorn` command: `--port 8001`
- Frontend: Update `vite.config.ts` port setting

### CORS Issues
CORS is configured to allow all origins in development. For production, update `backend/app/main.py` to specify allowed origins.



