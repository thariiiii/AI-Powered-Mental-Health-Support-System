from pydantic import BaseModel

class EmotionResponse(BaseModel):
    dominant_emotion: str
    confidence: str