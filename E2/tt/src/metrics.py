# E2/tt/src/metrics.py
import torch
from sentence_transformers import SentenceTransformer, util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    print(f"SBERT model 'all-MiniLM-L6-v2' loaded onto {DEVICE}.")
except Exception as e:
    print(f"Could not load SBERT model: {e}")
    SBERT_MODEL = None

def calculate_semantic_similarity(sentence1: str, sentence2: str) -> float:
    if SBERT_MODEL is None:
        return -1.0

    embeddings = SBERT_MODEL.encode([sentence1, sentence2], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
