from pydantic import BaseModel

class Exercise(BaseModel):
    id: str
    question: str
    answer: str
    difficulty: str
    topic: str
    created_at: str
    updated_at: str


