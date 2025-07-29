import os
import httpx
from dotenv import load_dotenv

load_dotenv()

RERANKER_URL = os.getenv("ULTRASAFE_RERANKER_URL")
RERANKER_MODEL = os.getenv("ULTRASAFE_RERANKER_MODEL", "usf1-rerank")
API_KEY = os.getenv("ULTRASAFE_API_KEY")

async def rerank(query: str, texts: list[str]) -> list[str]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": RERANKER_MODEL,
        "query": query,
        "texts": texts
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(RERANKER_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return [item["text"] for item in data["result"]["data"]]
