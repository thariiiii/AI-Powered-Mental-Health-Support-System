from fastapi import FastAPI
from app.routers import cbt, users, cbt_generate, cbt_evaluate
from app.services.distortion_classifier import DistortionClassifier
from app.services.domain_classifier import DomainClassifier
from app.services.rl_personalizer import RLPersonalizer

app = FastAPI(
    title="CBT Personalization Engine",
    description="A service for generating personalized Cognitive Behavioral Therapy exercises using classification, RL, and LLMs."
)

# --- Application Startup/Teardown ---
# Initialize models here to ensure they are singletons and loaded once.
@app.on_event("startup")
async def startup_event():
    # Load heavy models at startup
    app.state.distortion_clf = DistortionClassifier()
    app.state.domain_clf = DomainClassifier()
    app.state.rl_personalizer = RLPersonalizer()
    
    # Replace the placeholder in the router with the loaded instances
    cbt.distortion_clf = app.state.distortion_clf
    cbt.domain_clf = app.state.domain_clf
    cbt.rl_personalizer = app.state.rl_personalizer
    print("All AI models initialized and loaded.")

app.include_router(cbt.router, prefix="/api/v1/cbt")

app.include_router(users.router)

app.include_router(cbt_generate.router, prefix="/api/v2/cbt/generate")

app.include_router(cbt_evaluate.router, prefix="/api/v2/cbt/evaluate")

@app.get("/")
async def root():
    return {"message": "CBT Exercise Generator Service is running"}