"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import predict, explain, eda
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Financial Fraud Detection API",
    description="Explainable AI Model for Financial Fraud Detection Using ML",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(eda.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/api/predict/",
            "explainability": "/api/explain/",
            "eda": "/api/eda/",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

