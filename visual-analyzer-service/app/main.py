from fastapi import FastAPI
from app.routers import video
from app.routers import image  # <--- Import the new router

app = FastAPI(
    title="Visual Analyzer Service",
    description="Multi-model emotion detection service (YOLO + Keras)",
    version="1.1.0"
)

# 1. Register Video Router (Keras)
app.include_router(video.router, prefix="/api/v1/video", tags=["Video Analysis"])

# 2. Register Image Router (YOLO)
app.include_router(image.router, prefix="/api/v1/image", tags=["Image Analysis"])

@app.get("/")
async def root():
    return {
        "message": "Visual Analyzer Service is running",
        "endpoints": {
            "video_predict": "/api/v1/video/predict",
            "image_predict": "/api/v1/image/predict", 
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}