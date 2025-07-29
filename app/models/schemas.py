from pydantic import BaseModel
from typing import List


class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str


class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    topic: str

class TextRequest(BaseModel):
    text: str

class EntityExtractionRequest(BaseModel):
    text: str

class EntityExtractionResponse(BaseModel):
    entities: List[str]

class SentimentResponse(BaseModel):
    sentiment: str

