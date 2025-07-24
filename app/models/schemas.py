from pydantic import BaseModel
from typing import List, Optional

class SummarizeRequest(BaseModel):
    text: str
    webhook_url: Optional[str] = None

class SummarizeResponse(BaseModel):
    summary: str
