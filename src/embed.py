from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "documents.txt")
emb_path = os.path.join(BASE_DIR, "data", "embeddings.npy")

# Load documents
with open(data_path, 'r', encoding='utf-8') as f:
    documents = [line.strip() for line in f.readlines()]

# Generate embeddings
embeddings = model.encode(documents)

# Save embeddings
np.save(emb_path, embeddings)

print("Embeddings generated and saved!")