import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key="AIzaSyC1Xb1lc43Qrv2S4uqId8wL05aTpmY-WR0")

# Select model
MODEL = genai.GenerativeModel("gemini-2.5-flash")

EXERCISE_RULES = {
    "Behavioral Activation Plan": (
        "Ask the user to plan ONE small, concrete activity. "
        "The exercise must ask for: activity + time. "
        "The correct answer must be a short example plan."
    ),

    "Thought Record": (
        "Ask the user to identify ONE automatic negative thought and one alternative balanced thought. "
        "The correct answer must show a realistic example."
    ),

    "ABC Diary": (
        "Ask the user to write A (event), B (belief), and C (emotion). "
        "The correct answer must contain one short ABC example."
    ),

    "Positive Data Log": (
        "Ask the user to list one positive event and their role in it. "
        "The correct answer must be a short example."
    ),

    "Mindfulness Reflection": (
        "Ask the user to briefly describe what they noticed without judgment. "
        "The correct answer must show neutral observation language."
    )
}

class ExerciseGenerator:
    @staticmethod
    def check_available_models():
        """Check if the Gemini model is available."""
        try:
            models = genai.list_models()
            available = []
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    print(f"Available model: {m.name}")
                    available.append(m.name)
            return available # Return the list instead of just True for better debugging
        except Exception as e:
            print(f"Error checking models: {e}")
            return []

    @staticmethod
    def generate_exercise(exercise_type: str, user_history=None):

        history_text = ExerciseGenerator.format_history(user_history)

        exercise_rule = EXERCISE_RULES.get(
            exercise_type,
            "Generate a short CBT exercise with a brief example answer."
        )

        prompt = f"""
        You are a CBT therapist AI.

        Generate ONE CBT exercise only.

        EXERCISE TYPE:
        {exercise_type}

        USER HISTORY (context only):
        {history_text}

        RULES (MANDATORY):
        - The exercise must be SHORT (2–3 sentences max)
        - The exercise must ask the user to WRITE something
        - The correct_answer must be a SHORT example of a good user response
        - Do NOT include headings, explanations, encouragement, or therapy language
        - Do NOT teach CBT concepts
        - Do NOT use markdown
        - Do NOT include multiple steps

        EXERCISE-SPECIFIC RULE:
        {exercise_rule}

        OUTPUT FORMAT (JSON ONLY):
        {{
        "exercise": "...",
        "correct_answer": "..."
        }}
        """

        try:
            response = MODEL.generate_content(prompt)
            raw_output = response.text.strip()
            return ExerciseGenerator.extract_json(raw_output)

        except Exception as e:
            return {
                "exercise": "Unable to generate exercise.",
                "correct_answer": ""
            }


    @staticmethod
    def format_history(history):
        """Convert previous performance data into readable text."""
        if not history:
            return "No previous attempts available."

        text = ""
        for item in history:
            text += (
                f"- Previous exercise type: {item.get('exercise_type')}\n"
                f"  Score: {item.get('score')}\n"
                f"  Was Correct: {item.get('is_correct')}\n"
            )
        return text or "No previous history."

    @staticmethod
    def extract_json(text):
        """Extract JSON from the model output safely."""
        import json
        import re

        # Remove code block markers if present
        cleaned = re.sub(r"```json|```", "", text).strip()

        try:
            return json.loads(cleaned)
        except:
            # If parsing fails → fallback minimal output
            return {
                "exercise": "Could not parse exercise.",
                "correct_answer": ""
            }

    # ---------------------------------------------------------------------
    # GRADING SYSTEM
    # ---------------------------------------------------------------------

    @staticmethod
    def grade_response(exercise: str, correct_answer: str, user_response: str):
        """
        Grade user's CBT exercise response using Gemini.
        Evaluates conceptual correctness, not exact matching.
        Returns: score (0–1), feedback
        """

        prompt = f"""
        You are a CBT therapist AI evaluating a user's exercise response.

        IMPORTANT:
        - CBT exercises do NOT have one correct answer.
        - The user's response should be graded based on whether it
        correctly completes the CBT task described in the exercise.
        - Do NOT compare answers word-for-word.

        EXERCISE TASK:
        {exercise}

        EXAMPLE OF A GOOD RESPONSE (for reference only):
        {correct_answer}

        USER RESPONSE:
        {user_response}

        GRADING CRITERIA:
        - 1.0 → Fully completes the exercise correctly
        - 0.7 → Mostly correct, minor issues or missing detail
        - 0.4 → Partial attempt, key components missing
        - 0.0 → Did not attempt or response is unrelated

        ### TASK
        Respond ONLY in valid JSON:

        {{
        "score": <float between 0 and 1>,
        "feedback": "<short supportive CBT-style feedback>"
        }}
        """

        try:
            response = MODEL.generate_content(prompt)
            raw_output = response.text.strip()
            return ExerciseGenerator.extract_json(raw_output)

        except Exception as e:
            return {
                "score": 0.0,
                "feedback": "Unable to evaluate the response at this time."
            }