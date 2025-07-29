from fastapi import APIRouter
from app.services.nlp import summarize_text, classify_text, extract_entities, analyze_sentiment
from app.models.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    TextRequest,
    ClassificationResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
    SentimentResponse  
)

router = APIRouter()

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    summary = await summarize_text(request.text)
    return SummarizeResponse(summary=summary)

@router.post("/classify", response_model=ClassificationResponse)
async def classify(request: TextRequest):
    label = await classify_text(request.text)
    return ClassificationResponse(topic=label)

@router.post("/entities", response_model=EntityExtractionResponse)
async def extract_entities_route(request: EntityExtractionRequest):
    entities = await extract_entities(request.text)
    return EntityExtractionResponse(entities=entities)

@router.post("/sentiment", response_model=SentimentResponse)
async def sentiment_analysis(request: TextRequest):
    sentiment = await analyze_sentiment(request.text)
    return SentimentResponse(sentiment=sentiment)
