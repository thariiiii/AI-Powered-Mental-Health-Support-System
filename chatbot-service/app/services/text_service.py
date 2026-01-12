from pathlib import Path
from typing import Optional
import joblib
import neattext.functions as nfx
from fastapi import HTTPException

# --- Global Cache ---
_text_emotion_model: Optional[object] = None

def _get_model_path() -> Path:
    # Resolves to project_root/models/
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "models" / "emotion_classifier_pipe_lr.pkl"

def load_text_model() -> None:
    global _text_emotion_model
    if _text_emotion_model is not None:
        return

    model_path = _get_model_path()
    if not model_path.exists():
        raise HTTPException(status_code=500, detail=f"Text model not found at {model_path}")

    try:
        _text_emotion_model = joblib.load(open(model_path, "rb"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading text model: {exc}")

def predict_text_emotion(text: str) -> dict:
    load_text_model()
    
    # Preprocessing
    clean_text = nfx.remove_userhandles(text)
    clean_text = nfx.remove_stopwords(clean_text)
    
    # Prediction
    prediction = _text_emotion_model.predict([clean_text])[0]
    prediction_proba = _text_emotion_model.predict_proba([clean_text]).tolist()
    
    labels = _text_emotion_model.classes_.tolist()
    probabilities = dict(zip(labels, prediction_proba[0]))
    
    return {
        "original_text": text,
        "cleaned_text": clean_text,
        "predicted_emotion": prediction,
        "probabilities": probabilities,
    }