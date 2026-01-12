from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    app_name: str = "Auth Service"
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    db_url: str = "mongodb://localhost:27017/auth_db"  # Example for MongoDB

    class Config:
        env_file = "app/.env"

settings = Settings()