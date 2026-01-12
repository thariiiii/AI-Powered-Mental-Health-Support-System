import cv2
import numpy as np
import tensorflow as tf
import os

class EmotionAnalyzer:
    def __init__(self, model_path="emotion_model.keras"):
        # 1. Load the Model
        # We check if file exists to prevent crashing if path is wrong
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print("Loading Keras model...")
        self.model = tf.keras.models.load_model(model_path)
        
        # 2. Define Class Names (MUST match training alphabetical order)
        self.class_names = ['angry', 'neutral', 'sad'] 
        
        # 3. Load Face Detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_predictions = []
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 5th frame to speed up processing
            if frame_count % 5 == 0:
                # Convert to gray for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    # Crop face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Resize to 224x224 (Model Input Size)
                    face_img = cv2.resize(face_img, (224, 224))
                    
                    # Expand dimensions to match batch shape (1, 224, 224, 3)
                    img_array = tf.keras.utils.img_to_array(face_img)
                    img_array = tf.expand_dims(img_array, 0)
                    
                    # Predict
                    predictions = self.model.predict(img_array, verbose=0)
                    score = tf.nn.softmax(predictions[0])
                    
                    # Store the result (Index of highest probability)
                    predicted_class_index = np.argmax(score)
                    confidence = 100 * np.max(score)
                    
                    frame_predictions.append({
                        "emotion": self.class_names[predicted_class_index],
                        "confidence": confidence
                    })
            
            frame_count += 1
            
        cap.release()
        
        # Calculate Logic: Majority Vote (Which emotion appeared most?)
        if not frame_predictions:
            return "neutral", "0.0%" # Default if no face found

        emotions_found = [p['emotion'] for p in frame_predictions]
        
        # Find most frequent emotion
        dominant_emotion = max(set(emotions_found), key=emotions_found.count)
        
        # Calculate average confidence for that emotion
        relevant_confidences = [p['confidence'] for p in frame_predictions if p['emotion'] == dominant_emotion]
        avg_confidence = sum(relevant_confidences) / len(relevant_confidences)
        
        return dominant_emotion, f"{avg_confidence:.1f}%"

# Create a global instance
# Ensure 'emotion_model.keras' is in the root folder where you run main.py
# If it's in ml_models/, change path to "ml_models/emotion_model.keras"
analyzer = EmotionAnalyzer(model_path="emotion_model.keras")