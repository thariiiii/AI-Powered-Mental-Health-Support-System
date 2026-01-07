# ------------------------------------------------------------
# File: evaluate.py
# Description: Gemini-based CBT exercise evaluation service
# Author: Perera K.A.S.N
# ------------------------------------------------------------

import google.generativeai as genai
import re
import json

MODEL = genai.GenerativeModel("gemini-2.5-flash")


class CBTEvaluator:
    @staticmethod
    def evaluate(exercise: str, correct_answer: str, user_response: str):
        """
        Evaluate CBT response conceptually (not exact matching).
        Returns: score (0–1), is_correct (bool), feedback
        """

        prompt = f"""
        You are a CBT therapist AI evaluating a user's response.

        IMPORTANT RULES:
        - CBT exercises do NOT have exact correct answers.
        - Judge whether the user COMPLETED the CBT TASK.
        - Do NOT compare wording.
        - Be fair and supportive.

        EXERCISE:
        {exercise}

        EXAMPLE GOOD RESPONSE (reference only):
        {correct_answer}

        USER RESPONSE:
        {user_response}

        SCORING:
        - 1.0 = Fully correct
        - 0.7 = Mostly correct
        - 0.4 = Partial
        - 0.0 = Incorrect / unrelated

        OUTPUT JSON ONLY:
        {{
          "score": <float>,
          "is_correct": <true|false>,
          "feedback": "<short CBT-style feedback>"
        }}
        """

        try:
            response = MODEL.generate_content(prompt)
            return CBTEvaluator._extract_json(response.text)

        except Exception:
            return {
                "score": 0.0,
                "is_correct": False,
                "feedback": "Unable to evaluate response at this time."
            }

    @staticmethod
    def _extract_json(text: str):
        cleaned = re.sub(r"```json|```", "", text).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "score": 0.0,
                "is_correct": False,
                "feedback": "Evaluation parsing failed."
            }
