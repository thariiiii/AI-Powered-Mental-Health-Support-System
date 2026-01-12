import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import AudioFilePath
from app.services.speech_service import predict_speech_emotion

router = APIRouter(prefix="/speech-emotion", tags=["Speech Emotion"])

@router.post("/predict/filepath")
async def predict_from_filepath(audio_input: AudioFilePath):
    """Predict emotion from a local server-side file path."""
    return predict_speech_emotion(audio_input.file_path)

@router.post("/predict/uploadfile")
async def predict_from_upload(file: UploadFile = File(...)):
    """Predict emotion from an uploaded audio file."""
    temp_file_path = f"temp_{file.filename}"
    try:
        # Save upload to temp file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Process
        result = predict_speech_emotion(temp_file_path)
        
        # Cleanup
        os.remove(temp_file_path)
        return result
        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))