from fastapi import FastAPI
from app.routers import speech, text

app = FastAPI(title="Chatbot Emotion Service")

# Include Routers
app.include_router(speech.router, prefix="/chat")
app.include_router(text.router, prefix="/chat")

@app.get("/")
async def root():
    return {
        "message": "Chatbot Emotion Service is running.",
        "endpoints": [
            # "/chat/speech-emotion/predict/filepath",
            "/chat/speech-emotion/predict/uploadfile",
            "/chat/text-emotion/predict"
        ]
    }