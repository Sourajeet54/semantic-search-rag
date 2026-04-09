import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# Load data
documents = open("data/documents.txt", "r", encoding="utf-8").read().split("\n")
embeddings = np.load("data/embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("Semantic Search RAG (Offline)")
query = st.text_input("Ask a question")

if query:
    query_embedding = model.encode([query])
    scores = np.dot(embeddings, query_embedding.T).flatten()
    top_idx = np.argmax(scores)
    context = documents[top_idx]

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "Answer based on context."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
        ]
    )

    st.write(response['message']['content'])