from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class AudioFilePath(BaseModel):
    file_path: str