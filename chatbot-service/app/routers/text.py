from fastapi import APIRouter
from app.schemas import TextRequest
from app.services.text_service import predict_text_emotion

router = APIRouter(prefix="/text-emotion", tags=["Text Emotion"])

@router.post("/predict")
async def predict_text(request: TextRequest):
    """Predict emotion from raw text input."""
    return predict_text_emotion(request.text)