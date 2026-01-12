import os
import numpy as np
from stable_baselines3 import DQN

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../ml_models/rl_emotion_intervention_dqn.zip")

try:
    RL_MODEL = DQN.load(MODEL_PATH)
    print("[RL] Emotion Intervention Model Loaded")
except Exception as e:
    print("[RL] Load failed:", e)
    RL_MODEL = None


class RLPersonalizer:
    INTERVENTIONS = [
        "Guided Breathing",
        "Grounding Technique",
        "Positive Affirmation",
        "Mindful Reflection",
        "Gratitude Prompt"
    ]

    @staticmethod
    def select_intervention(emotion: str, intensity: str, context: str, engagement: float, success: float):
        # Encode string inputs into numerical state
        emotion_map = {"anxiety": 0.1, "sadness": 0.3, "anger": 0.6, "stress": 0.8, "neutral": 0.5}
        intensity_map = {"low": 0.2, "medium": 0.5, "high": 0.9}
        context_map = {"work": 0.2, "relationships": 0.5, "health": 0.8, "self": 0.4}

        state = np.array([
            emotion_map.get(emotion.lower(), 0.5),
            intensity_map.get(intensity.lower(), 0.5),
            context_map.get(context.lower(), 0.5),
            engagement,
            success
        ], dtype=np.float32)

        if RL_MODEL is None:
            # return np.random.choice(RLPersonalizer.INTERVENTIONS)
            return {"message": "RL model not loaded"}

        action, _ = RL_MODEL.predict(state, deterministic=True)
        return RLPersonalizer.INTERVENTIONS[action]
