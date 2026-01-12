from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.image_emotion_service import image_analyzer

router = APIRouter()

# --- Response Schemas ---
class PredictionOut(BaseModel):
    emotion: str
    confidence: float
    all_confidences: List[float]

class PredictionsResponse(BaseModel):
    predictions: List[PredictionOut]

# --- Endpoint ---
@router.post("/predict", response_model=PredictionsResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotions from an uploaded image using YOLOv11.
    """
    # 1. Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image. Accepted formats: jpeg, png, gif, bmp, webp"
        )

    try:
        # 2. Call the YOLO Service
        # We pass the file directly to the service
        raw_predictions = await image_analyzer.detect_emotions(file)
        
        if not raw_predictions:
            return PredictionsResponse(predictions=[])
            # Or raise 400 if you prefer to error when no face is found:
            # raise HTTPException(status_code=400, detail="No emotions detected")
        
        # 3. Format Response
        prediction_objects = [
            PredictionOut(
                emotion=p["emotion"],
                confidence=round(p["confidence"], 4), # Rounding for cleaner output
                all_confidences=p["all_confidences"]
            )
            for p in raw_predictions
        ]
        
        return PredictionsResponse(predictions=prediction_objects)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Emotion detection failed: {str(e)}"
        )