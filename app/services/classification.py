import os
import httpx
from app.services.embedding import get_embedding
from app.services.reranker import rerank
from app.database.vector_store_helper import get_vector_store, save_vector_store

TOPIC_LABELS = [
    "politics", "finance", "health", "technology", "sports",
    "entertainment", "science", "education", "environment", "travel"
]

ULTRASAFE_API_KEY = os.getenv("ULTRASAFE_API_KEY")
ULTRASAFE_API_URL = os.getenv("ULTRASAFE_API_URL")
ULTRASAFE_MODEL = os.getenv("ULTRASAFE_MODEL")


async def classify_text(text: str) -> str:
    # 1. Get embedding
    embedding = await get_embedding(text)

    # 2. Retrieve from vector store
    vector_store = get_vector_store()
    retrieved = vector_store.search(embedding, top_k=5)

    if not retrieved:
        context = ""
    else:
        summaries = [item["summary"] for item in retrieved]
        reranked = await rerank(text, summaries)

        retrieved_filtered = [item for item in retrieved if item["summary"] in reranked]
        reranked_items = sorted(retrieved_filtered, key=lambda x: reranked.index(x["summary"]))
        context = "\n".join(item["summary"] for item in reranked_items[:3])

    # 3. Formulate classification prompt
    label_list = ", ".join(TOPIC_LABELS)
    prompt = f"""
    Context:
    {context}

    Input:
    {text}

    Task:
    Classify the input into one of the following topics:
    {label_list}

    Respond with only the topic name.
    """

    headers = {
        "Authorization": f"Bearer {ULTRASAFE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": ULTRASAFE_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(ULTRASAFE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        topic = response.json()["choices"][0]["message"]["content"].strip()

    # 4. Add to vector store
    vector_store.add(embedding, text, topic, task_type="classification")
    save_vector_store()

    return topic
