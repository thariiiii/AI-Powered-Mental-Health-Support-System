from fastapi import APIRouter, HTTPException
from app.services.evaluate import CBTEvaluator
from app.services import db_service
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(tags=["CBT Evaluation"])


class AutoEvaluateRequest(BaseModel):
    user_id: str
    exercise_id: str
    user_response: str


@router.post("/auto-evaluate")
async def auto_evaluate_cbt(req: AutoEvaluateRequest):
    user_id = req.user_id
    exercise_id = req.exercise_id
    user_response = req.user_response

    # 1️⃣ Fetch user
    user = db_service.get_user(user_id)
    sessions = user.get("CBTExerciseSessions", [])

    # 2️⃣ Find session
    session = next(
        (s for s in sessions if isinstance(s, dict) and s.get("exercise_id") == exercise_id),
        None
    )

    if not session:
        raise HTTPException(status_code=404, detail="CBT session not found")

    exercise = session["exercise"]
    correct_answer = session.get("correct_answer", "")

    # 3️⃣ Evaluate using Gemini
    evaluation = CBTEvaluator.evaluate(
        exercise=exercise,
        correct_answer=correct_answer,
        user_response=user_response
    )

    score = evaluation["score"]
    is_correct = evaluation["is_correct"]

    # 4️⃣ Update DB
    db_service.update_cbt_session_feedback(
        user_id=user_id,
        exercise_id=exercise_id,
        feedback_data={
            "user_response": user_response,
            "score": score,
            "is_correct": is_correct,
            "feedback": evaluation.get("feedback"),
            "evaluated_at": datetime.utcnow()
        }
    )

    # 5️⃣ Response
    return {
        "exercise_id": exercise_id,
        "score": score,
        "is_correct": is_correct,
        "feedback": evaluation.get("feedback")
    }
