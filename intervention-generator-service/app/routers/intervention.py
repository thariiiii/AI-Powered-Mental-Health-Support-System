from fastapi import APIRouter
from app.services.rl_personalizer import RLPersonalizer

router = APIRouter()

@router.post("/intervention")
def get_intervention(data: dict):
    result = RLPersonalizer.select_intervention(
        emotion=data.get("emotion", "neutral"),
        intensity=data.get("intensity", "medium"),
        context=data.get("context", "self"),
        engagement=data.get("engagement", 0.5),
        success=data.get("success", 0.5)
    )
    return {"recommended_intervention": result}
