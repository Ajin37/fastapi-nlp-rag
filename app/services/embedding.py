import os
import httpx
from typing import List

ULTRASAFE_EMBEDDING_URL = os.getenv("ULTRASAFE_EMBEDDING_URL")
ULTRASAFE_EMBEDDING_MODEL = os.getenv("ULTRASAFE_EMBEDDING_MODEL")
ULTRASAFE_API_KEY = os.getenv("ULTRASAFE_API_KEY")


async def get_embedding(text: str, model: str = ULTRASAFE_EMBEDDING_MODEL) -> List[float]:
    payload = {
        "model": model,
        "input": text,
    }

    headers = {
        "Authorization": f"Bearer {ULTRASAFE_API_KEY}"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(ULTRASAFE_EMBEDDING_URL, json=payload, headers=headers)
        response.raise_for_status()

        try:
            embedding = response.json()["result"]["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected embedding response format: {response.text}") from e

        return embedding
