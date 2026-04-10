from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# load documents
documents = open("data/documents.txt", "r", encoding="utf-8").read().split("\n")

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load FAISS index
index = faiss.read_index("data/faiss.index")

def search(query, k=3):
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = [documents[i] for i in indices[0]]
    return results