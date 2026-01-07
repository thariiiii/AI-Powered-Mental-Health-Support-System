from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid

from app.services.distortion_classifier import DistortionClassifier
from app.services.domain_classifier import DomainClassifier
from app.services.rl_personalizer import RLPersonalizer
from app.services.exercise_generator import ExerciseGenerator
from app.services import db_service

# router = APIRouter(prefix="/cbt", tags=["CBT Auto"])

router = APIRouter()

distortion_clf = DistortionClassifier()
domain_clf = DomainClassifier()
rl_personalizer = RLPersonalizer()
exercise_generator = ExerciseGenerator()


@router.post("/auto-generate")
async def auto_generate_cbt_exercise(user_id: str):
    """
    Auto CBT Exercise Generation:
    - Uses last input
    - Auto classifies missing fields
    - Uses RL for exercise selection
    - Uses DB history
    """

    # ------------------ 1. Fetch User ------------------
    try:
        user = db_service.get_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    inputs = user.get("inputs", [])
    if not inputs:
        raise HTTPException(status_code=400, detail="No user inputs available")

    last_input = inputs[-1]

    user_message = last_input.get("user_message", "")

    # ------------------ 2. Distortion ------------------
    if not last_input.get("detected_distortion_cbt"):
        distortion_result = distortion_clf.predict(user_message)
        last_input["detected_distortion_cbt"] = distortion_result["top_distortion"]
        db_service.update_last_input(user_id, last_input)

    top_distortion = last_input["detected_distortion_cbt"]

    # ------------------ 3. Domain ------------------
    if not last_input.get("detected_domain_cbt"):
        domain_result = domain_clf.predict(user_message)
        last_input["detected_domain_cbt"] = domain_result["domain"]
        db_service.update_last_input(user_id, last_input)

    top_domain = last_input["detected_domain_cbt"]

    # ------------------ 4. Emotion ------------------
    emotion_state = "Neutral"
    if user.get("current_emotion"):
        emotion_state = user["current_emotion"].get("emotion", "Neutral")

    # ------------------ 5. Engagement & Success ------------------
    engagement = db_service.calculate_engagement(user)
    success = db_service.calculate_success(user)

    # ------------------ 6. RL Personalization ------------------
    exercise_type = rl_personalizer.select_exercise(
        distortion=top_distortion,
        emotion=emotion_state,
        engagement=engagement,
        success=success,
        domain=top_domain
    )

    # ------------------ 7. History ------------------
    last_session = db_service.get_last_cbt_session(user)
    history = [last_session] if last_session else []

    # ------------------ 8. Generate Exercise ------------------
    llm_result = exercise_generator.generate_exercise(
        exercise_type=exercise_type,
        user_history=history
    )

    exercise_text = llm_result.get("exercise")
    correct_answer = llm_result.get("correct_answer", "")

    if not exercise_text:
        raise HTTPException(status_code=500, detail="Exercise generation failed")

    # ------------------ 9. Store CBT Session ------------------
    exercise_id = str(uuid.uuid4())

    session = {
        "exercise_id": exercise_id,
        "recommended_exercise_type": exercise_type,
        "exercise": exercise_text,
        "correct_answer": correct_answer,
        "is_correct": False,
        "timestamp": datetime.utcnow()
    }

    db_service.store_cbt_session(user_id, session)

    # ------------------ 10. Response ------------------
    return {
        "exercise_id": exercise_id,
        "exercise_type": exercise_type,
        "exercise": exercise_text,
        "distortion": top_distortion,
        "domain": top_domain,
        "emotion": emotion_state
    }
