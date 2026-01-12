from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import EmotionResponse
from app.services.emotion_service import analyzer
import shutil
import os
import uuid

router = APIRouter()

@router.post("/predict", response_model=EmotionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.MOV')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video.")

    # 2. Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Call the Service to analyze
        # Note: This is synchronous processing. For heavy production use, 
        # you would use BackgroundTasks or Celery.
        emotion, confidence = analyzer.analyze_video(temp_filename)
        
        return {
            "dominant_emotion": emotion,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Cleanup: Delete temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)