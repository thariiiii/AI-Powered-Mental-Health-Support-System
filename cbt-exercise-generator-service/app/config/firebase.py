import firebase_admin
from firebase_admin import credentials, firestore
import os

# Path to Firebase service account key (stored in project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIREBASE_CRED_PATH = os.path.join(BASE_DIR, "firebase-service-account-key.json")

# Validate credential file existence
if not os.path.exists(FIREBASE_CRED_PATH):
    raise FileNotFoundError(
        "firebase-service-account-key.json not found in project root"
    )

# Initialize Firebase App (Singleton)
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)

# Firestore client (used across app)
db = firestore.client()
