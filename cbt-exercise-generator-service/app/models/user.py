from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Sub-models ---
class Inputs(BaseModel):
    user_message: str
    detected_emotion: Optional[str] = None
    detected_distortion_cbt: Optional[str] = None
    detected_domain_cbt: Optional[str] = None
    source: Optional[str] = None #text/voice
    timestamp: datetime

class CBTExerciseSession(BaseModel):
    exercise_id: str
    recommended_exercise_type: str
    exercise: str
    correct_answer: str
    user_response: str
    score: Optional[float] = None
    user_rating: Optional[int] = None
    is_correct: bool
    timestamp: datetime

class CurrentEmotion(BaseModel):
    emotion: str
    source: str #text/voice/image/video
    timestamp: datetime

# --- Main User Model ---
class User(BaseModel):
    user_id: str
    email: EmailStr
    full_name: Optional[str] = None
    inputs: List[Inputs] = []
    current_emotion: Optional[CurrentEmotion] = None
    CBTExerciseSessions: List[CBTExerciseSession] = []
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime

# Request body for /cbt/generate
class GenerateExerciseRequest(BaseModel):
    user_id: str
    user_message: str

# Response for /cbt/generate (The exercise is the primary output)
class GenerateExerciseResponse(BaseModel):
    exercise_id: str
    recommended_exercise_type: str
    exercise: str
    top_distortion: str
    top_domain: str
    emotion: Optional[str] = None

# Request body for /cbt/submit
class SubmitExerciseRequest(BaseModel):
    user_id: str
    exercise_id: str
    user_response: str
    user_rating: Optional[int] = None # e.g., 1-5 for how helpful it was