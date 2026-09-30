from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model import detector

app = FastAPI(title="Job Fraud Detection Microservice", version="1.0")

class JobPosting(BaseModel):
    id: str
    title: str
    company: str
    description: str
    email: Optional[str] = ""

class BatchPredictionRequest(BaseModel):
    jobs: List[JobPosting]

class PredictionResponse(BaseModel):
    job_id: str
    is_fraudulent: bool
    confidence: float

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Microservicio de IA operando correctamente"}

@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest):
    try:
        jobs_data = [job.dict() for job in payload.jobs]
        results = detector.predict_batch(jobs_data)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))