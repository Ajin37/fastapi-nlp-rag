import os
import httpx
from typing import List
import re
import json

from app.services.embedding import get_embedding
from app.services.reranker import rerank
from app.database.vector_store_helper import get_vector_store, save_vector_store

ULTRASAFE_API_KEY = os.getenv("ULTRASAFE_API_KEY")
ULTRASAFE_API_URL = os.getenv("ULTRASAFE_API_URL")
ULTRASAFE_MODEL = os.getenv("ULTRASAFE_MODEL")

async def summarize_text(text: str) -> str:
    embedding = await get_embedding(text)
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

    prompt = f"""
    Context:
    {context}

    Input:
    {text}

    Task:
    Summarize the input using the given context if it's relevant.
    """

    headers = {
        "Authorization": f"Bearer {ULTRASAFE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": ULTRASAFE_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(ULTRASAFE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"].strip()

    vector_store.add(embedding, text, summary, task_type="summarization")
    save_vector_store()
    return summary

async def classify_text(text: str) -> str:
    embedding = await get_embedding(text)
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

    categories = [
        "finance", "health", "technology", "education",
        "sports", "politics", "entertainment", "science", "travel", "environment"
    ]

    prompt = f"""
    Context:
    {context}

    Input:
    {text}

    Task:
    Classify the input into one of the following general topic categories:
    {", ".join(categories)}

    Return only the category name.
    """

    headers = {
        "Authorization": f"Bearer {ULTRASAFE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": ULTRASAFE_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(ULTRASAFE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        label = response.json()["choices"][0]["message"]["content"].strip()

    vector_store.add(embedding, text, label, task_type="classification")
    save_vector_store()
    return label

async def extract_entities(text: str) -> List[str]:
    embedding = await get_embedding(text)
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

    prompt = f"""
    Context:
    {context}

    Input:
    {text}

    Task:
    Extract all named entities (people, organizations, locations, products, etc.) mentioned in the input.
    Return the entities as a Python list of strings.
    Example format: ["Apple", "Tim Cook", "California"]
    """

    headers = {
        "Authorization": f"Bearer {ULTRASAFE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": ULTRASAFE_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(ULTRASAFE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        entities_raw = response.json()["choices"][0]["message"]["content"].strip()

    cleaned = re.sub(r"```(?:\w+)?", "", entities_raw).replace("```", "").strip()

    try:
        entities = json.loads(cleaned)
        if not isinstance(entities, list):
            entities = [cleaned]
    except Exception:
        entities = [cleaned]

    vector_store.add(embedding, text, ", ".join(entities), task_type="entity_extraction")
    save_vector_store()
    return entities

async def analyze_sentiment(text: str) -> str:
    embedding = await get_embedding(text)
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

    prompt = f"""
    Context:
    {context}

    Input:
    {text}

    Task:
    Analyze the sentiment of the input text.
    Return only one of these labels: Positive, Negative, or Neutral.
    """

    headers = {
        "Authorization": f"Bearer {ULTRASAFE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": ULTRASAFE_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(ULTRASAFE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        sentiment = response.json()["choices"][0]["message"]["content"].strip()

    vector_store.add(embedding, text, sentiment, task_type="sentiment_analysis")
    save_vector_store()
    return sentiment
