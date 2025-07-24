from fastapi import APIRouter
from app.models.schemas import SummarizeRequest, SummarizeResponse

router = APIRouter()

@router.get("/")
def root():
    return {"status": "API running"}

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    # This will be implemented later
    return {"summary": f"Mock summary of: {request.text[:50]}..."}
