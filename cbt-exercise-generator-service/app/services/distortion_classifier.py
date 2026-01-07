from transformers import pipeline

class DistortionClassifier:
    def __init__(self):
        # Load the zero-shot pipeline
        # "facebook/bart-large-mnli" is robust for zero-shot text classification
        print("[DistortionClassifier] Loading model... (this may take a moment)")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Standard CBT Cognitive Distortions
        self.labels = [
            "All-or-Nothing Thinking",
            "Catastrophizing",
            "Overgeneralization",
            "Mental Filtering",
            "Disqualifying the Positive",
            "Jumping to Conclusions",
            "Magnification or Minimization",
            "Emotional Reasoning",
            "Should Statements",
            "Labeling and Mislabeling",
            "Personalization"
        ]

    def predict(self, text, multi_label=True):
        """
        Classifies the text into one or more cognitive distortions.
        multi_label=True allows a sentence to have multiple distortions (common in CBT).
        """
        result = self.classifier(text, candidate_labels=self.labels, multi_label=multi_label)
        
        # Return the top label and its score
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        return {
            "top_distortion": top_label,
            "confidence": top_score,
            "all_predictions": dict(zip(result['labels'], result['scores']))
        }

# Usage Example (for testing)
if __name__ == "__main__":
    classifier = DistortionClassifier()
    text = "I failed this one exam, so I'm going to fail my entire degree and never get a job."
    print(classifier.predict(text))