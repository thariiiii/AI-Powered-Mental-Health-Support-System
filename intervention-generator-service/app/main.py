from fastapi import FastAPI
from app.routers import intervention

app = FastAPI(title="Adaptive Intervention Generator")
app.include_router(intervention.router, prefix="/api/v1/intervention")

@app.get("/")
async def root():
    return {"message": "Adaptive Intervention Generator Service is running"}