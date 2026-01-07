from fastapi import APIRouter
from app.services.rl_personalizer import RLPersonalizer
from app.services.distortion_classifier import DistortionClassifier
from app.services.domain_classifier import DomainClassifier
from app.services.exercise_generator import ExerciseGenerator
from app.services import db_service
from app.models.user import GenerateExerciseRequest, GenerateExerciseResponse, SubmitExerciseRequest
import uuid
import json
from datetime import datetime

router = APIRouter()
personalizer = RLPersonalizer()
distortion_classifier = DistortionClassifier()
domain_classifier = DomainClassifier()
generator = ExerciseGenerator()

# ---------------------------------------------------------------------
# Dummy/Test CBT Router Endpoints (No DB Integration)
# ---------------------------------------------------------------------

@router.post("/personalize")
def personalize_exercise(data: dict):
    exercise = personalizer.select_exercise(
        distortion=data.get("distortion"),
        emotion=data.get("emotion"),
        engagement=data.get("engagement"),
        success=data.get("success"),
        domain=data.get("domain"),
    )
    return {"recommended_exercise": exercise}

@router.post("/distortion")
def classify_distortion(data: dict):
    distortion = distortion_classifier.predict(data.get("text", ""))
    return distortion

@router.post("/domain")
def classify_domain(data: dict):
    domain = domain_classifier.predict(data.get("text", ""))
    return domain 

@router.post("/available/genai_models")
def available_genai_models():
    models = generator.check_available_models()
    return models

@router.post("/generate/dummy")
async def generate_cbt_exercise():
    dummy_history = [
        {
            "exercise_type": "Thought Record",
            "score": 0.45,
            "is_correct": False,
            "note": "User identified the automatic thought but struggled to find balanced alternative thoughts."
        },
        {
            "exercise_type": "ABC Diary",
            "score": 0.6,
            "is_correct": True,
            "note": "User correctly identified activating event and belief but consequences lacked emotional detail."
        },
        {
            "exercise_type": "Positive Data Log",
            "score": 0.75,
            "is_correct": True,
            "note": "User recorded positive events but minimized their personal contribution."
        },
        {
            "exercise_type": "Mindfulness Reflection",
            "score": 0.85,
            "is_correct": True,
            "note": "User demonstrated good present-moment awareness with minimal judgment."
        }
    ]

    generate = generator.generate_exercise(
        exercise_type="Behavioral Activation Plan",  # Placeholder (future RL output)
        user_history=dummy_history
    )

    exercise_id = str(uuid.uuid4())
    return generate

@router.post("/evaluate/dummy")
async def evaluate_cbt_exercise_dummy():
    """
    Dummy endpoint to test CBT exercise evaluation.
    """

    # Dummy exercise (what the user was asked to do)
    exercise = (
        "Write one negative thought you had today and then write a more balanced alternative thought."
    )

    # Example of a good response (NOT the only correct answer)
    correct_answer = (
        "Negative thought: I always fail. "
        "Balanced thought: I have failed before, but I have also succeeded."
    )

    # Dummy user response (simulated user input)
    user_response = (
        "I thought I was terrible at my job, but then I remembered I’ve done well on past tasks."
    )

    # Grade the response
    evaluation = ExerciseGenerator.grade_response(
        exercise=exercise,
        correct_answer=correct_answer,
        user_response=user_response
    )

    return {
        "exercise": exercise,
        "user_response": user_response,
        "evaluation": evaluation
    }

# ---------------------------------------------------------------------
# Main CBT Router Endpoints With Database Integration (Not yet complete)
# ---------------------------------------------------------------------

# Initialize models (singleton approach)
# These should be loaded once at application startup (in main.py)
# But for simplicity, we'll initialize them here (might lead to slow startup)
try:
    distortion_clf = DistortionClassifier()
    domain_clf = DomainClassifier()
    rl_personalizer = RLPersonalizer()
except Exception as e:
    # Handle the case where models (especially RL) fail to load
    print(f"Error loading AI models: {e}")
    distortion_clf = None
    domain_clf = None
    rl_personalizer = None
    
# Use a simple placeholder for the exercise generator instance
exercise_generator = ExerciseGenerator()

@router.post("/generate", response_model=GenerateExerciseResponse)
async def generate_cbt_exercise(req: GenerateExerciseRequest):
    """
    1. Classify the user message (Distortion, Domain).
    2. Fetch user history/state.
    3. Use RL model to select the best exercise type.
    4. Generate the personalized exercise using the LLM.
    5. Store initial session data.
    """
    
    user_id = req.user_id
    user_message = req.user_message

    if not distortion_clf or not domain_clf or not rl_personalizer:
        raise HTTPException(status_code=503, detail="AI classification/personalization services are not available.")
        
    # 0. Set a unique ID for the session
    exercise_id = str(uuid.uuid4())

    # --- 1. Classification ---
    try:
        distortion_result = distortion_clf.predict(user_message, multi_label=False)
        domain_result = domain_clf.predict(user_message)
        
        top_distortion = distortion_result['top_distortion']
        top_domain = domain_result['domain']
        
    except Exception as e:
        # Fallback to defaults if classification fails
        print(f"Classification failed: {e}")
        top_distortion = "Overgeneralization"
        top_domain = "Stress"
    
    # Store the input and the classifications for history
    input_data = {
        "user_message": user_message,
        "detected_emotion": "Unknown", # Emotion detection would typically be another step
        "detected_distortion_cbt": top_distortion,
        "detected_domain_cbt": top_domain,
        "source": "text",
        "timestamp": datetime.now()
    }
    try:
        db_service.update_user_input(user_id, input_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        pass # Ignore minor DB update failure and continue

    # --- 2. Fetch User State for RL ---
    try:
        user_state = db_service.fetch_user_data(user_id)
        
        # RL requires a current emotion, so we use the fetched state or a default
        emotion_state = user_state.get("current_emotion", "Neutral")
        # RL also requires engagement/success
        engagement = user_state.get("engagement_count", 1) # simple count
        success = user_state.get("average_success_score", 0.5) # average score
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        # Fallback state if DB fails
        emotion_state = "Neutral"
        engagement = 1
        success = 0.5
        user_state = {"latest_input": None}
    
    # --- 3. RL Personalization (Select Exercise) ---
    recommended_type = rl_personalizer.select_exercise(
        distortion=top_distortion,
        emotion=emotion_state,
        engagement=engagement,
        success=success,
        domain=top_domain
    )
    
    # --- 4. Generate Exercise ---
    # The LLM prompt is better with the original user message
    history_for_llm = [
        {"exercise_type": recommended_type, "score": success, "is_correct": success >= 0.8}
    ] 

    llm_result = exercise_generator.generate_exercise(
        exercise_type=recommended_type,
        user_history=[user_state["latest_input"] or {}] # Pass the latest input as a proxy for 'history'
    )
    
    exercise_text = llm_result.get("exercise", "Error: Could not generate exercise.")
    correct_answer = llm_result.get("correct_answer", "")
    
    if "Error" in exercise_text:
         raise HTTPException(status_code=500, detail=exercise_text)


    # --- 5. Store Initial Session Data ---
    session_data = {
        "user_id": user_id,
        "exercise_id": exercise_id,
        "recommended_exercise_type": recommended_type,
        "exercise": exercise_text,
        "correct_answer": correct_answer,
        # Store initial predictions for context
        "initial_distortion": top_distortion,
        "initial_domain": top_domain
    }
    
    db_service.store_cbt_exercise_session(user_id, session_data, exercise_id)

    # --- 6. Response ---
    return GenerateExerciseResponse(
        exercise_id=exercise_id,
        recommended_exercise_type=recommended_type,
        exercise=exercise_text,
        top_distortion=top_distortion,
        top_domain=top_domain,
        emotion=emotion_state
    )

@router.post("/submit")
async def submit_exercise_feedback(req: SubmitExerciseRequest):
    """
    1. Fetch the original exercise details.
    2. Grade the user response using the LLM.
    3. Calculate the RL reward.
    4. Store the final session data (response, score, rating).
    5. Update the RL agent's policy (retrain or store experience).
    """
    
    user_id = req.user_id
    exercise_id = req.exercise_id
    user_response = req.user_response
    user_rating = req.user_rating
    
    # --- 1. Fetch Original Exercise ---
    # We must fetch the original session from the DB to get the exercise text and correct answer
    try:
        session_ref = db_service.db.collection("sessions").document(exercise_id).get()
        if not session_ref.exists:
            raise HTTPException(status_code=404, detail="Exercise session not found.")
        session_data = session_ref.to_dict()
        
        exercise_text = session_data["exercise"]
        correct_answer = session_data["correct_answer"]
        recommended_type = session_data["recommended_exercise_type"]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching session: {e}")

    # --- 2. Grade User Response ---
    llm_grade = exercise_generator.grade_response(
        exercise=exercise_text,
        correct_answer=correct_answer,
        user_response=user_response
    )
    
    score = llm_grade.get("score", 0.0)
    feedback_text = llm_grade.get("feedback", "No feedback available.")
    
    # --- 3. Calculate RL Reward ---
    # A simple reward function: combine the grading score and the user's rating.
    # The reward should be a float between 0 and 1.
    
    # Example: 70% from score, 30% from user rating (scaled 1-5 to 0-1)
    if user_rating is None: user_rating = 3 # Neutral default
    scaled_rating = (user_rating - 1) / 4.0 # Scales 1->0, 5->1
    
    rl_reward = (score * 0.7) + (scaled_rating * 0.3)
    
    # --- 4. Store Final Session Data ---
    final_feedback = {
        "user_response": user_response,
        "score": score,
        "user_rating": user_rating,
        "feedback": feedback_text
    }
    
    db_service.update_session_with_feedback(user_id, exercise_id, final_feedback, rl_reward)
    
    # --- 5. Update RL Agent ---
    if rl_personalizer:
        # In a real system, we'd store the (state, action, reward) tuple in the DB
        # and periodically retrain. For this example, we log it.
        rl_personalizer.update_from_feedback(rl_reward)
    
    # --- 6. Response ---
    return {
        "message": "Exercise submitted and graded successfully.",
        "score": score,
        "feedback": feedback_text,
        "reward_for_rl": rl_reward
    }
