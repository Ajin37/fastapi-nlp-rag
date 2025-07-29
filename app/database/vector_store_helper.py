# app/database/vector_store_helper.py

from app.database.vector_store import InMemoryVectorStore

EMBEDDING_DIM = 1024

# Singleton instance
_vector_store: InMemoryVectorStore = None

def get_vector_store() -> InMemoryVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = InMemoryVectorStore(dim=EMBEDDING_DIM)
        try:
            _vector_store.load("vector_store_data")
            print("[VectorStore] Loaded from disk.")
        except Exception as e:
            print("[VectorStore] Starting fresh:", e)
    return _vector_store

def save_vector_store():
    if _vector_store is not None:
        _vector_store.save("vector_store_data")
        print("[VectorStore] Saved to disk.")
