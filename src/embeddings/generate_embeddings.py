import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/processed/unified_catalog_clean_tags.csv"
OUTPUT_DATA_PATH = "data/processed/unified_catalog_with_embeddings.csv"
OUTPUT_EMB_PATH = "data/processed/unified_catalog_embeddings.npy"

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

texts = df["search_text"].fillna("").astype(str).tolist()

print(f"Loaded {len(texts)} datasets")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings)

np.save(OUTPUT_EMB_PATH, embeddings)

df["embedding_model"] = model_name
df.to_csv(OUTPUT_DATA_PATH, index=False)

print("Embeddings shape:", embeddings.shape)
print("Saved metadata:", OUTPUT_DATA_PATH)
print("Saved embeddings:", OUTPUT_EMB_PATH)