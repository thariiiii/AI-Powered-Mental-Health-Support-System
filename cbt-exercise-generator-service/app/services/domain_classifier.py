from transformers import pipeline

class DomainClassifier:
    def __init__(self):
        # We can reuse the same model instance if optimizing for memory, 
        # but for clarity, we initialize it here.
        print("[DomainClassifier] Loading model...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Broad Mental Health Domains
        self.labels = [
            "Depression",
            "Anxiety",
            "Self-Esteem",
            "Stress",
            "Social Isolation",
            "Anger Management",
            "Grief"
        ]

    def predict(self, text):
        """
        Classifies the mental health domain. 
        multi_label=False assumes the text focuses on one primary domain.
        """
        result = self.classifier(text, candidate_labels=self.labels, multi_label=False)
        
        return {
            "domain": result['labels'][0],
            "confidence": result['scores'][0],
            "all_predictions": dict(zip(result['labels'], result['scores']))
        }

# Usage Example
if __name__ == "__main__":
    clf = DomainClassifier()
    print(clf.predict("I feel like everyone is watching me and judging me whenever I go out."))