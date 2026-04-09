from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "documents.txt")
emb_path = os.path.join(BASE_DIR, "data", "embeddings.npy")

# Load data
with open(data_path, 'r', encoding='utf-8') as f:
    documents = [line.strip() for line in f.readlines()]

embeddings = np.load(emb_path)

# Query
query = input("Enter your query: ")

# Encode query
query_embedding = model.encode([query])

# Similarity
similarities = cosine_similarity(query_embedding, embeddings)[0]

# Top 3 results
top_k = 3
top_indices = similarities.argsort()[-top_k:][::-1]

print("\nTop matches:")
for idx in top_indices:
    print("-", documents[idx])