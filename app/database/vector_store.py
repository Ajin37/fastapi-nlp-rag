import os
import pickle
import faiss
import numpy as np
from typing import List

class InMemoryVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.summaries = []
        self.task_types = []

    def add(self, vector: List[float], text: str, summary: str, task_type: str):
        vector_np = np.array(vector).astype("float32").reshape(1, -1)
        assert vector_np.shape[1] == self.dim, f"Embedding dimension {vector_np.shape[1]} does not match index dimension {self.dim}"
        self.index.add(vector_np)
        self.texts.append(text)
        self.summaries.append(summary)
        self.task_types.append(task_type)

    def search(self, vector: List[float], top_k: int = 5):
        vector_np = np.array(vector).astype("float32").reshape(1, -1)
        D, I = self.index.search(vector_np, top_k)
        results = []

        for idx in I[0]:
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "summary": self.summaries[idx],
                "task_type": self.task_types[idx],
                "distance": D[0][list(I[0]).index(idx)]
            })

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "summaries": self.summaries,
                "task_types": self.task_types
            }, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.summaries = data["summaries"]
            self.task_types = data["task_types"]

        if self.index.ntotal != len(self.texts):
            print(f"[VectorStore] Mismatch: index has {self.index.ntotal}, texts has {len(self.texts)}")
            print("[VectorStore] Deleting corrupt index. Starting fresh.")
            os.remove(os.path.join(path, "index.faiss"))
            os.remove(os.path.join(path, "data.pkl"))
            self.index = faiss.IndexFlatL2(self.dim)
            self.texts = []
            self.summaries = []
            self.task_types = []
