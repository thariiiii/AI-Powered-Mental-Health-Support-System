# # services/db_service.py
# import firebase_admin
# from firebase_admin import credentials, firestore
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# from app.models.user import Inputs, CBTExerciseSession # Import necessary models

# # Initialize Firebase (assuming a standard setup as in the original file)
# load_dotenv()
# try:
#     # Use the existing credentials setup
#     if not firebase_admin._apps:
#         cred = credentials.Certificate("firebase-service-account-key.json")
#         firebase_admin.initialize_app(cred)
# except Exception as e:
#     print(f"Firebase initialization failed: {e}")
#     # Handle the error appropriately in a production environment

# db = firestore.client()

# # --- Existing Functions ---

# def fetch_user_data(user_id: str):
#     # ... (Keep existing fetch_user_data)
#     user_ref = db.collection("users").document(user_id)
#     user = user_ref.get()
#     if not user.exists:
#         raise ValueError(f"User with ID '{user_id}' not found")
#     user_data = user.to_dict()
    
#     # Simple formatting for the RL agent
#     feedback_scores = [s for s in user_data.get("feedback_scores", []) if s is not None]
    
#     # We will fetch the latest input message from the 'inputs' subcollection/list
#     latest_input = user_data.get("inputs")[-1] if user_data.get("inputs") else None
    
#     return {
#         "latest_input": latest_input,
#         "current_emotion": user_data.get("current_emotion", {}).get("emotion"),
#         "average_success_score": sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.5,
#         "engagement_count": len(user_data.get("CBTExerciseSessions", []))
#     }

# # --- New/Modified Functions ---

# def update_user_input(user_id: str, input_data: dict):
#     """Stores the new input from the user (message, detected features)"""
#     user_ref = db.collection("users").document(user_id)
#     input_model = Inputs(**input_data)
#     user_ref.update({
#         "inputs": firestore.ArrayUnion([input_model.dict()])
#     })

# def store_cbt_exercise_session(user_id: str, session_data: dict, exercise_id: str):
#     """
#     Stores the full CBT exercise session data (exercise, model prediction, but no user answer yet).
#     Returns the document ID.
#     """
#     session_data["timestamp"] = firestore.SERVER_TIMESTAMP
    
#     # 1. Store in the sessions collection for detailed tracking
#     session_ref = db.collection("sessions").document(exercise_id)
#     session_ref.set(session_data)
    
#     # 2. Add a reference/summary to the user's document
#     user_ref = db.collection("users").document(user_id)
    
#     # Only store the necessary parts in the user document to avoid bloat
#     summary_data = {
#         "exercise_id": exercise_id,
#         "recommended_exercise_type": session_data.get("recommended_exercise_type"),
#         "timestamp": session_data.get("timestamp")
#     }
    
#     user_ref.update({
#         "CBTExerciseSessions": firestore.ArrayUnion([summary_data])
#     })
    
#     return exercise_id

# def update_session_with_feedback(user_id: str, exercise_id: str, feedback: dict, rl_reward: float):
#     """Updates a session with user response, score, and user rating."""
    
#     # 1. Update the session in the 'sessions' collection
#     session_ref = db.collection("sessions").document(exercise_id)
#     update_data = {
#         "user_response": feedback["user_response"],
#         "score": feedback["score"],
#         "user_rating": feedback["user_rating"],
#         "is_correct": feedback["score"] >= 0.8, # Simple threshold for 'correct'
#         "feedback_text": feedback["feedback"],
#         "rl_reward": rl_reward # For future RL retraining
#     }
#     session_ref.update(update_data)
    
#     # 2. Update the user's feedback scores for RL state
#     user_ref = db.collection("users").document(user_id)
    
#     # Also update the summary in the user's CBTExerciseSessions list (optional, but good for completeness)
#     # This part is more complex and might involve a transaction for safety in a real app,
#     # but for simplicity, we focus on the score array and session update.
#     user_ref.update({
#         "feedback_scores": firestore.ArrayUnion([feedback.get("score")])
#     })

# app/services/db_service.py

from google.cloud.firestore_v1 import ArrayUnion
from datetime import datetime
from typing import Dict, Any
from app.config.firebase import db  # your Firestore init

# ------------------ USER ------------------

def get_user(user_id: str) -> Dict[str, Any]:
    ref = db.collection("users").document(user_id).get()
    if not ref.exists:
        raise ValueError("User not found")
    return ref.to_dict()

def update_last_input(user_id: str, updated_input: Dict[str, Any]):
    user_ref = db.collection("users").document(user_id)
    user = user_ref.get().to_dict()
    inputs = user.get("inputs", [])
    inputs[-1] = updated_input
    user_ref.update({"inputs": inputs, "updated_at": datetime.utcnow()})

# ------------------ CBT ------------------

def store_cbt_session(user_id: str, session: dict):
    user_ref = db.collection("users").document(user_id)

    user_ref.update({
        "CBTExerciseSessions": ArrayUnion([session]),
        "updated_at": datetime.utcnow()
    })

def get_last_cbt_session(user: Dict[str, Any]) -> Dict[str, Any]:
    sessions = user.get("CBTExerciseSessions", [])
    return sessions[-1] if sessions else None

def calculate_engagement(user: Dict[str, Any]) -> int:
    return len(user.get("CBTExerciseSessions", []))

def calculate_success(user: Dict[str, Any]) -> float:
    sessions = user.get("CBTExerciseSessions", [])
    if not sessions:
        return 0.5
    scores = [s.get("score", 0) for s in sessions if s.get("score") is not None]
    return sum(scores) / len(scores) if scores else 0.5

def update_cbt_session_feedback(
    user_id: str,
    exercise_id: str,
    feedback_data: dict
):
    ref = db.collection("users").document(user_id)
    user = ref.get().to_dict()

    sessions = user.get("CBTExerciseSessions", [])
    if not isinstance(sessions, list):
        raise ValueError("CBTExerciseSessions is corrupted")

    updated = False
    for s in sessions:
        if isinstance(s, dict) and s.get("exercise_id") == exercise_id:
            s.update(feedback_data)
            updated = True
            break

    if not updated:
        raise ValueError("CBT session not found")

    ref.update({"CBTExerciseSessions": sessions})
