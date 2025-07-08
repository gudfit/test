import torch
from sentence_transformers import SentenceTransformer, util

# --- Global Initialization ---
# Load the SBERT model once and reuse it to save time and memory.
# 'all-MiniLM-L6-v2' is a great, fast, and effective choice.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    print(f"SBERT model 'all-MiniLM-L6-v2' loaded onto {DEVICE}.")
except Exception as e:
    print(f"Could not load SBERT model: {e}")
    SBERT_MODEL = None
# --- End Global Initialization ---

def calculate_semantic_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculates the semantic similarity between two sentences using SBERT.
    Returns -1.0 on failure to load model.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: A cosine similarity score between 0 and 1, or -1.0 if model not loaded.
    """
    if SBERT_MODEL is None:
        return -1.0

    # The model can take a list of sentences. Encoding them together is efficient.
    embeddings = SBERT_MODEL.encode([sentence1, sentence2], convert_to_tensor=True)
    
    # Compute cosine-similarity
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
    
    return cosine_score.item()

# --- Self-Testing Block ---
if __name__ == "__main__":
    print("\n--- Running metrics.py self-test ---")
    if SBERT_MODEL is not None:
        s1 = "The cat sat on the mat."
        s2 = "A feline was resting on the rug."
        s3 = "The weather is sunny today."

        sim_1_2 = calculate_semantic_similarity(s1, s2)
        sim_1_3 = calculate_semantic_similarity(s1, s3)

        print(f"Similarity ('{s1}', '{s2}'): {sim_1_2:.4f}")
        print(f"Similarity ('{s1}', '{s3}'): {sim_1_3:.4f}")

        assert sim_1_2 > 0.8, "High similarity expected between s1 and s2"
        assert sim_1_3 < 0.3, "Low similarity expected between s1 and s3"
        print("\nSemantic similarity self-test completed successfully.")
    else:
        print("SBERT model not loaded. Cannot run self-test.")
