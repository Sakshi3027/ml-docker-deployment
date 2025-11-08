"""
FastAPI application for serving the ML model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pickle
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A machine learning API for sentiment analysis",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = "models/sentiment_model.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("⚠ Model file not found. Please train the model first.")
    model = None

# Request/Response Models
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product, it's amazing!"
            }
        }

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This is great!",
                    "I don't like this",
                    "It's okay, nothing special"
                ]
            }
        }

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for a single text
    
    - **text**: The text to analyze (1-1000 characters)
    
    Returns the predicted sentiment (Positive/Negative) with confidence score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        prediction = model.predict([input_data.text])[0]
        probabilities = model.predict_proba([input_data.text])[0]
        
        # Get sentiment and confidence
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            text=input_data.text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts
    
    - **texts**: List of texts to analyze (1-100 items)
    
    Returns predictions for all texts
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        # Make predictions for all texts
        pred_labels = model.predict(input_data.texts)
        pred_probs = model.predict_proba(input_data.texts)
        
        for text, label, probs in zip(input_data.texts, pred_labels, pred_probs):
            sentiment = "Positive" if label == 1 else "Negative"
            confidence = float(max(probs))
            
            predictions.append(PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=round(confidence, 4),
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)