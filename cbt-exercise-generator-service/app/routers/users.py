# routers/users.py
from fastapi import APIRouter, HTTPException
from app.services.db_service import db
from app.models.user import User, Inputs, CurrentEmotion, CBTExerciseSession
from datetime import datetime, timedelta
import random
import uuid

router = APIRouter(prefix="/users", tags=["User Management"])

# --- Request Model ---
from pydantic import BaseModel, EmailStr
class InitUserRequest(BaseModel):
    email: EmailStr
    full_name: str = "Test User"
    user_id: str = None # Optional: if not provided, one will be generated

@router.post("/init", response_model=dict)
async def create_initial_user(req: InitUserRequest):
    """
    Creates a user with dummy data including:
    - 3 past inputs (User messages)
    - 3 completed CBT sessions (Simulating history for RL)
    - Current emotion state
    """
    
    # 1. Determine User ID
    user_id = req.user_id if req.user_id else str(uuid.uuid4())
    
    # 2. Check if exists
    doc_ref = db.collection("users").document(user_id)
    if doc_ref.get().exists:
        raise HTTPException(status_code=400, detail="User ID already exists.")

    # 3. Generate Dummy Timestamps (Past 3 days)
    now = datetime.now()
    t1 = now - timedelta(days=3)
    t2 = now - timedelta(days=2)
    t3 = now - timedelta(days=1)

    # 4. Dummy Inputs (History of what they said)
    dummy_inputs = [
        Inputs(
            user_message="I feel overwhelmed by my workload and can't focus.",
            detected_emotion="Anxious",
            detected_distortion_cbt="Catastrophizing",
            detected_domain_cbt="Stress",
            source="text",
            timestamp=t1
        ),
        Inputs(
            user_message="My friend didn't text back, they probably hate me.",
            detected_emotion="Sad",
            detected_distortion_cbt="Jumping to Conclusions",
            detected_domain_cbt="Social Isolation",
            source="text",
            timestamp=t2
        ),
        Inputs(
            user_message="I'm useless because I made one mistake.",
            detected_emotion="Depressed",
            detected_distortion_cbt="All-or-Nothing Thinking",
            detected_domain_cbt="Self-Esteem",
            source="text",
            timestamp=t3
        )
    ]

    # 5. Dummy CBT Sessions (History of what they did)
    # This provides the RL agent with "Engagement" and "Success" history
    dummy_sessions = [
        CBTExerciseSession(
            exercise_id=str(uuid.uuid4()),
            recommended_exercise_type="Breathing Exercise",
            exercise="Take deep breaths...",
            correct_answer="N/A",
            user_response="I did it.",
            score=0.9, # High success
            user_rating=5,
            is_correct=True,
            timestamp=t1
        ),
        CBTExerciseSession(
            exercise_id=str(uuid.uuid4()),
            recommended_exercise_type="Thought Record",
            exercise="Identify the trigger...",
            correct_answer="The trigger was...",
            user_response="I don't know.",
            score=0.4, # Low success
            user_rating=2,
            is_correct=False,
            timestamp=t2
        ),
        CBTExerciseSession(
            exercise_id=str(uuid.uuid4()),
            recommended_exercise_type="Cognitive Restructuring",
            exercise="Challenge the thought...",
            correct_answer="Evidence against...",
            user_response="They might be busy.",
            score=0.85, # High success
            user_rating=4,
            is_correct=True,
            timestamp=t3
        )
    ]

    # 6. Construct User Object
    new_user = User(
        user_id=user_id,
        email=req.email,
        full_name=req.full_name,
        inputs=dummy_inputs,
        current_emotion=CurrentEmotion(
            emotion="Anxious",
            source="text",
            timestamp=now
        ),
        CBTExerciseSessions=dummy_sessions,
        is_active=True,
        created_at=now,
        updated_at=now
    )

    # 7. Save to Firestore
    # Note: .dict() is used for Pydantic v1, .model_dump() for v2.
    # We use jsonable_encoder style dict conversion to ensure datetime compatibility if needed,
    # but Firestore client usually handles python datetime objects well.
    try:
        doc_ref.set(new_user.dict()) 
    except AttributeError:
        doc_ref.set(new_user.model_dump()) # Pydantic v2 fallback

    return {
        "message": "User created successfully with dummy data.",
        "user_id": user_id,
        "history_count": len(dummy_inputs),
        "session_count": len(dummy_sessions)
    }