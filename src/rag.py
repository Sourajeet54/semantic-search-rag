from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import ollama

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "documents.txt")
emb_path = os.path.join(BASE_DIR, "data", "embeddings.npy")

# Load documents
with open(data_path, 'r', encoding='utf-8') as f:
    documents = [line.strip() for line in f.readlines()]

embeddings = np.load(emb_path)

# Query
query = input("Ask your question: ")

# Encode query
query_embedding = model.encode([query])

# Similarity search
similarities = cosine_similarity(query_embedding, embeddings)[0]

# Top retrieval
top_k = 3
top_indices = similarities.argsort()[-top_k:][::-1]
context = "\n".join([documents[i] for i in top_indices])

# Short-answer prompt
prompt = f"""
Answer briefly in 2-3 sentences using only the context below.

Context:
{context}

Question:
{query}
"""

# Local LLM call
response = ollama.chat(
    model='llama3',
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nAnswer:")
print(response['message']['content'])