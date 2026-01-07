import os
import numpy as np
from stable_baselines3 import DQN

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../ml_models/rl_personalizer_dqn.zip")

# Load RL model once
try:
    RL_MODEL = DQN.load(MODEL_PATH)
    print("[RL] Model loaded successfully from", MODEL_PATH)
except Exception as e:
    print("[RL] Failed to load model:", e)
    RL_MODEL = None


class RLPersonalizer:
    """RL agent wrapper for selecting CBT exercise types."""

    EXERCISES = [
        "Thought Record",
        "Socratic Questioning",
        "Behavioral Activation Plan",
        "Positive Data Log",
        "ABC Diary",
        "Cognitive Restructuring",
        "Mindfulness Reflection",
        "Gratitude Journal",
        "Exposure Hierarchy",
        "Behavioral Experiment"
    ]

    DISTORTION_MAP = {
        "All-or-Nothing Thinking": 0.0,
        "Catastrophizing": 0.1,
        "Overgeneralization": 0.2,
        "Mind Reading": 0.3,
        "Emotional Reasoning": 0.4,
        "Labeling": 0.5,
        "Personalization": 0.6,
        "Mental Filtering": 0.7,
        "Should Statements": 0.8,
        "Fortune Telling": 0.9
    }

    EMOTION_MAP = {
        "Happy": 0.0, "Sad": 0.1, "Angry": 0.2, "Anxious": 0.3, "Fearful": 0.4,
        "Stressed": 0.5, "Calm": 0.6, "Lonely": 0.7, "Frustrated": 0.8, "Hopeful": 0.9
    }

    DOMAIN_MAP = {
        "Depression": 0.0, "Anxiety": 0.2, "Self-Esteem": 0.4,
        "Stress Management": 0.6, "Social Skills": 0.8
    }

    def __init__(self):
        self.last_state = None
        self.last_action = None

    def encode_category(self, category: str, mapping: dict) -> float:
        """Encode a categorical label into a numeric value between 0–1."""
        if not category:
            return 0.5  # neutral default
        return mapping.get(category.strip().title(), 0.5)

    def select_exercise(self, distortion, emotion, engagement, success, domain):
        """Predict best CBT exercise using RL policy."""
        if RL_MODEL is None:
            return np.random.choice(self.EXERCISES)

        # Encode categorical text into numeric values
        distortion_val = self.encode_category(distortion, self.DISTORTION_MAP)
        emotion_val = self.encode_category(emotion, self.EMOTION_MAP)
        domain_val = self.encode_category(domain, self.DOMAIN_MAP)

        try:
            engagement_val = float(engagement)
            success_val = float(success)
        except ValueError:
            engagement_val, success_val = 0.5, 0.5

        # Create numeric state vector
        state = np.array([
            distortion_val, emotion_val, engagement_val, success_val, domain_val
        ], dtype=np.float32)

        # Save for feedback
        self.last_state = state

        # Predict best exercise type
        action, _ = RL_MODEL.predict(state, deterministic=True)
        self.last_action = action

        return self.EXERCISES[action]

    
    def update_from_feedback(self, reward):
        """Update RL model using the last (state, action, reward)."""
        if self.last_state is None or self.last_action is None:
            print("[RL] No previous state/action to update from.")
            return

        experience = (self.last_state, self.last_action, reward)
        print(f"[RL] Received feedback -> reward: {reward}, action: {self.last_action}")

        # store this in DB and retrain periodically
        # db.collection("rl_feedback").add({
        #     "state": self.last_state.tolist(),
        #     "action": int(self.last_action),
        #     "reward": float(reward)
        # })

    def retrain_from_experiences(self, experiences):
        """Re-train the RL model on new experiences periodically."""
        # Retraing logic
        pass
