import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib
import neattext.functions as nfx


router = APIRouter()


# --- Global variables for models and label encoders ---
_speech_emotion_model: Optional[object] = None
_speech_label_encoder: Optional[LabelEncoder] = None
_text_emotion_model: Optional[object] = None


# Define the known emotion labels based on the dataset structure.
# These should match the folders in your 'emotional_speech_set_data'.
KNOWN_OAF_LABELS = sorted(
    [
        "OAF_Fear",
        "OAF_Pleasant_surprise",
        "OAF_happy",
        "OAF_Sad",
        "OAF_angry",
        "OAF_disgust",
        "OAF_neutral",
    ]
)


def _get_model_path() -> Path:
    """
    Resolve the path to the emotional speech recognition model file
    relative to the `app` package.
    """
    base_dir = Path(__file__).resolve().parents[1]  # app/
    return base_dir / "models" / "emotional_speech_recognition_model.keras"


def _get_text_model_path() -> Path:
    """
    Resolve the path to the text emotion classifier model file
    relative to the `app` package.
    """
    base_dir = Path(__file__).resolve().parents[1]  # app/
    return base_dir / "models" / "emotion_classifier_pipe_lr.pkl"


def _load_speech_resources() -> None:
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

    _speech_emotion_model = load_model(str(model_path))

    encoder = LabelEncoder()
    encoder.fit(KNOWN_OAF_LABELS)
    _speech_label_encoder = encoder


def _load_text_emotion_model() -> None:
    """Load the trained text emotion classifier (lazy-loaded)."""
    global _text_emotion_model

    if _text_emotion_model is not None:
        return

    model_path = _get_text_model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Text emotion model file not found at {model_path}.",
        )

    try:
        _text_emotion_model = joblib.load(open(model_path, "rb"))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Error loading text emotion model: {exc}",
        ) from exc


def _extract_audio_features(file_path: str) -> np.ndarray:
    """Extract MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(file_path, res_type="kaiser_fast")
        features = np.mean(
            librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0
        )
        return features
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=400, detail=f"Error processing audio file: {e}"
        ) from e


def _predict_speech_emotion_core(audio_file_path: str) -> dict:
    """Core prediction logic for speech emotion recognition."""
    _load_speech_resources()
    assert _speech_emotion_model is not None
    assert _speech_label_encoder is not None

    features = _extract_audio_features(audio_file_path)
    features = features[np.newaxis, np.newaxis, :]

    predicted_probabilities = _speech_emotion_model.predict(features)
    predicted_label_index = int(np.argmax(predicted_probabilities))
    predicted_emotion_raw = _speech_label_encoder.classes_[predicted_label_index]

    # Emotion mapping for TESS dataset
    emotion_mapping = {
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

    recognizable_emotion = emotion_mapping.get(
        predicted_emotion_raw, predicted_emotion_raw
    )
    return {"predicted_emotion": recognizable_emotion}


class TextRequest(BaseModel):
    text: str


def _predict_text_emotion_core(text: str) -> dict:
    """Core prediction logic for text-based emotion recognition."""
    _load_text_emotion_model()
    assert _text_emotion_model is not None

    # Preprocess the input text using the same steps as training
    clean_text = nfx.remove_userhandles(text)
    clean_text = nfx.remove_stopwords(clean_text)

    # Make prediction
    prediction = _text_emotion_model.predict([clean_text])[0]
    prediction_proba = _text_emotion_model.predict_proba([clean_text]).tolist()

    # Get all possible labels and their probabilities
    labels = _text_emotion_model.classes_.tolist()
    probabilities = dict(zip(labels, prediction_proba[0]))

    return {
        "original_text": text,
        "cleaned_text": clean_text,
        "predicted_emotion": prediction,
        "probabilities": probabilities,
    }


class AudioFilePath(BaseModel):
    file_path: str


@router.post("/speech-emotion/predict/filepath")
async def predict_speech_emotion_from_path(audio_input: AudioFilePath):
    """
    Predict emotion from a local audio file path (server-side path).
    """
    return _predict_speech_emotion_core(audio_input.file_path)


@router.post("/text-emotion/predict")
async def predict_text_emotion(request: TextRequest):
    """
    Predict emotion from raw text input.

    Endpoint URL (including main app prefix): `/chat/text-emotion/predict`
    """
    return _predict_text_emotion_core(request.text)


@router.post("/speech-emotion/predict/uploadfile")
async def predict_speech_emotion_from_upload(file: UploadFile = File(...)):
    """
    Predict emotion from an uploaded audio file.
    """
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        result = _predict_speech_emotion_core(temp_file_path)
        os.remove(temp_file_path)
        return result
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e)) from e