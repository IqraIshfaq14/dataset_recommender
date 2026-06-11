import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/unified_catalog_with_embeddings.csv"
EMB_PATH = "data/processed/unified_catalog_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRecommender:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)
        self.embeddings = np.load(EMB_PATH)
        self.model = SentenceTransformer(MODEL_NAME)

    def recommend(self, query, top_k=10):
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        scores = np.dot(self.embeddings, query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.df.iloc[top_indices].copy()
        results["similarity_score"] = scores[top_indices]

        columns = [
            "dataset_id",
            "title",
            "description",
            "task",
            "downloads",
            "likes",
            "url",
            "similarity_score",
        ]

        available_columns = [col for col in columns if col in results.columns]
        return results[available_columns]


if __name__ == "__main__":
    recommender = EmbeddingRecommender()

    while True:
        query = input("\nEnter a dataset query (or 'quit'): ")

        if query.lower() == "quit":
            break

        results = recommender.recommend(query, top_k=10)

        print(f"\nQuery: {query}\n")

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            print("=" * 80)
            print(f"Rank #{rank}")
            print(f"Dataset: {row['dataset_id']}")
            print(f"Score: {row['similarity_score']:.4f}")

            if pd.notna(row.get("downloads")):
                print(f"Downloads: {row['downloads']}")

            if pd.notna(row.get("likes")):
                print(f"Likes: {row['likes']}")

            description = str(row.get("description", ""))

            if description and description != "nan":
                print("\nDescription:")
                print(
                    description[:250]
                    + ("..." if len(description) > 250 else "")
                )

            if pd.notna(row.get("url")):
                print(f"\nURL: {row['url']}")

            print()