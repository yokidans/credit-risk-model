from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from pathlib import Path
import uuid
import time
import logging
from datetime import datetime

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

app = FastAPI(
    title="Bati Bank Credit Scoring API",
    description="API for credit risk assessment",
    version="1.0.0"
)

# Simple in-memory cache (replace with proper caching in production)
prediction_cache = {}

class CreditRequest(BaseModel):
    customer_id: str
    features: Dict[str, Any]
    requested_amount: float = 0.0
    requested_duration: int = 0  # in days

class CreditResponse(BaseModel):
    customer_id: str
    probability_default: float
    credit_score: int
    approved: bool
    approved_amount: Optional[float] = None
    approved_duration: Optional[int] = None
    interest_rate: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Credit Risk API</title>
        </head>
        <body>
            <h1>Credit Risk Assessment API</h1>
            <p>API is running successfully!</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/redoc">Alternative Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """

@app.post("/assess_credit", response_model=CreditResponse)
async def assess_credit(request: CreditRequest):
    try:
        # In a real implementation, you would call your model here
        # This is a mock implementation
        mock_result = {
            "customer_id": request.customer_id,
            "probability_default": 0.15,
            "credit_score": 680,
            "approved": True,
            "approved_amount": request.requested_amount * 0.8,
            "approved_duration": request.requested_duration,
            "interest_rate": 5.5
        }
        return CreditResponse(**mock_result)
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)