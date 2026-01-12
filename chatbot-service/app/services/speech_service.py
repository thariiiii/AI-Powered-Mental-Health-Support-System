import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import HTTPException
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# --- Global Cache for Models & Encoders ---
_speech_emotion_model: Optional[object] = None
_speech_label_encoder: Optional[LabelEncoder] = None

# --- CONSTANTS ---

# CRITICAL FIX: The encoder must know ALL labels the model was trained on.
# These match the keys in your EMOTION_MAPPING.
ALL_TESS_LABELS = sorted(
    [
        "OAF_Fear",
        "OAF_Pleasant_surprised",
        "OAF_Sad",
        "OAF_angry",
        "OAF_disgust",
        "OAF_happy",
        "OAF_neutral",
        "YAF_fear",
        "YAF_pleasant_surprised",
        "YAF_sad",
        "YAF_angry",
        "YAF_disgust",
        "YAF_happy",
        "YAF_neutral",
    ]
)

# Mapping specific TESS dataset labels to general emotions
EMOTION_MAPPING = {
    "YAF_angry": "ANGRY",
    "YAF_disgust": "DISGUST",
    "YAF_fear": "FEAR",
    "YAF_happy": "HAPPY",
    "YAF_neutral": "NEUTRAL",
    "YAF_pleasant_surprised": "SURPRISED",
    "YAF_sad": "SAD",
    "OAF_angry": "ANGRY",
    "OAF_disgust": "DISGUST",
    "OAF_Fear": "FEAR",
    "OAF_happy": "HAPPY",
    "OAF_neutral": "NEUTRAL",
    "OAF_Pleasant_surprised": "SURPRISED",
    "OAF_Sad": "SAD",
}


def _get_model_path() -> Path:
    """
    Resolve the path to the emotional speech recognition model file.
    Assumes structure:
      project_root/
        app/services/speech_service.py
        models/emotional_speech_recognition_model.keras
    """
    # Go up 2 levels: services -> app -> project_root
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "models" / "emotional_speech_recognition_model.keras"


def load_speech_resources() -> None:
    """Load the trained speech emotion model and label encoder (lazy-loaded)."""
    global _speech_emotion_model, _speech_label_encoder

    if _speech_emotion_model is not None and _speech_label_encoder is not None:
        return

    model_path = _get_model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Speech emotion model file not found at {model_path}.",
        )

    # Load Keras Model
    try:
        _speech_emotion_model = load_model(str(model_path))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load Keras model: {e}"
        )

    # Fit Encoder with ALL known labels (Fixes the Index Out of Bounds error)
    encoder = LabelEncoder()
    encoder.fit(ALL_TESS_LABELS)
    _speech_label_encoder = encoder


def _extract_audio_features(file_path: str) -> np.ndarray:
    """Extract MFCC features from an audio file."""
    try:
        # Load audio with librosa (kaiser_fast is faster for resampling)
        audio, sr = librosa.load(file_path, res_type="kaiser_fast")
        
        # Extract MFCCs and take the mean across time axis
        features = np.mean(
            librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0
        )
        return features
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing audio file: {e}"
        )


def predict_speech_emotion(audio_file_path: str) -> dict:
    """Core prediction logic for speech emotion recognition."""
    
    # 1. Ensure resources are loaded
    load_speech_resources()
    assert _speech_emotion_model is not None
    assert _speech_label_encoder is not None

    # 2. Extract Features
    features = _extract_audio_features(audio_file_path)
    
    # 3. Reshape for LSTM/CNN input (Batch Size, 1, Features) or (Batch, Features, 1)
    # Note: Adjust dimensions based on your specific model training shape.
    # Common shapes are (1, 13, 1) or (1, 1, 13).
    # Based on your previous code, you used:
    features = features[np.newaxis, np.newaxis, :] 

    # 4. Predict
    predicted_probabilities = _speech_emotion_model.predict(features)
    predicted_label_index = int(np.argmax(predicted_probabilities))
    
    # 5. Decode Label
    try:
        predicted_emotion_raw = _speech_label_encoder.classes_[predicted_label_index]
    except IndexError:
        # Fallback if model predicts an index outside our label list (rare if list is correct)
        raise HTTPException(
            status_code=500, 
            detail=f"Model predicted index {predicted_label_index}, but encoder only has {len(_speech_label_encoder.classes_)} classes."
        )

    # 6. Map to simple emotion name
    final_emotion = EMOTION_MAPPING.get(predicted_emotion_raw, predicted_emotion_raw)

    return {
        "predicted_emotion": final_emotion,
        "raw_label": predicted_emotion_raw,
        "confidence": float(np.max(predicted_probabilities))
    }