from fastapi import FastAPI, Request, Response, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from src.nlp import DiseasePredictor
from src.utils import generate_html_response
from src.database import fetch_single_soap_note, update_ai_analysis
from src.config import get_config
import logging
import bleach

logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

app = FastAPI(
    title="Clinical NLP API",
    description="Real-time clinical NLP processing service for analyzing SOAP notes.",
    version="1.0.0"
)

def get_user_key(request: Request) -> str:
    """Get rate limit key based on user ID or remote address."""
    return request.headers.get("X-User-ID", get_remote_address(request))

limiter = Limiter(key_func=get_user_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions globally."""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def get_disease_predictor() -> DiseasePredictor:
    """Get a DiseasePredictor instance."""
    return DiseasePredictor()

class PredictionRequest(BaseModel):
    text: str
    department: Optional[str] = None

class ProcessNoteRequest(BaseModel):
    note_id: int

class DiseasePrediction(BaseModel):
    disease: str
    score: float

class EntityContext(BaseModel):
    severity: int
    temporal: str

class EntityDetail(BaseModel):
    text: str
    label: str
    context: EntityContext

class PredictionResponse(BaseModel):
    primary_diagnosis: Optional[DiseasePrediction]
    differential_diagnoses: List[DiseasePrediction]
    processing_time: float

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict diseases from clinical text",
    description="Analyzes clinical text to identify potential diseases and their confidence scores."
)
@limiter.limit(HIMS_CONFIG["RATE_LIMIT"])
async def predict(request: Request, payload: PredictionRequest, predictor: DiseasePredictor = Depends(get_disease_predictor)):
    """Predict diseases from clinical text."""
    import time
    start_time = time.time()
    cleaned_text = bleach.clean(payload.text)
    predictions = predictor.predict_from_text(cleaned_text)
    return PredictionResponse(
        primary_diagnosis=predictions["primary_diagnosis"],
        differential_diagnoses=predictions["differential_diagnoses"],
        processing_time=time.time() - start_time
    )

@app.post(
    "/process_note",
    summary="Process a SOAP note",
    description="Processes a SOAP note by ID, generating an HTML report with disease predictions and entities."
)
@limiter.limit(HIMS_CONFIG["RATE_LIMIT"])
async def process_note(request: Request, payload: ProcessNoteRequest, predictor: DiseasePredictor = Depends(get_disease_predictor)):
    """Process a SOAP note by ID."""
    import time
    start_time = time.time()
    note = fetch_single_soap_note(payload.note_id)
    if not note:
        return Response(
            content=generate_html_response({"detail": "Note not found"}, 404),
            status_code=404,
            media_type="text/html"
        )
    
    result = predictor.process_soap_note(note)
    if "error" in result:
        return Response(
            content=generate_html_response({"detail": result["error"]}, 400),
            status_code=400,
            media_type="text/html"
        )
    
    html_content = generate_html_response(result, 200)
    update_ai_analysis(note["id"], html_content, result['summary'])
    
    return Response(
        content=html_content,
        status_code=200,
        media_type="text/html"
    )

def start_server():
    """Start the FastAPI server."""
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host=HIMS_CONFIG["API_HOST"], port=HIMS_CONFIG["API_PORT"], log_level="info")