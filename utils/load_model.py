import os
from sentence_transformers import SentenceTransformer, util

def load_model_and_threshold():
    # Path to the local SBERT model directory
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "sbert_quora_model"))

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"❌ Model folder not found at: {model_dir}")

    # Load model
    model = SentenceTransformer(model_dir)

    # Load best threshold
    threshold_path = os.path.join(model_dir, "best_threshold.txt")
    if not os.path.isfile(threshold_path):
        raise FileNotFoundError(f"❌ Threshold file not found at: {threshold_path}")

    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())

    # Cosine similarity function
    def similarity_fn(q1, q2):
        emb1 = model.encode(q1, convert_to_tensor=True)
        emb2 = model.encode(q2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))

    return model, threshold, similarity_fn
