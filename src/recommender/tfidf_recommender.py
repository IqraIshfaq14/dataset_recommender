import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/unified_catalog_clean_tags.csv")
df = df.fillna("")

df["corpus"] = (
    df["title"] + " " +
    df["description"] + " " +
    df["tags"]
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

tfidf_matrix = vectorizer.fit_transform(df["corpus"])


def recommend_by_title(query_title, top_n=5):
    matches = df[df["title"].str.contains(query_title, case=False, na=False)]

    if matches.empty:
        print("No dataset found.")
        return

    idx = matches.index[0]

    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_indices = similarities.argsort()[::-1][1:top_n + 1]

    print(f"\nRecommendations for: {df.loc[idx, 'title']}\n")

    for rank, i in enumerate(similar_indices, start=1):
        print(f"{rank}. {df.loc[i, 'title']}")
        print(f"   Source: {df.loc[i, 'source']}")
        print(f"   Similarity: {similarities[i]:.4f}")
        print(f"   URL: {df.loc[i, 'url']}\n")


query = input("Enter dataset name: ")

recommend_by_title(query, top_n=5)