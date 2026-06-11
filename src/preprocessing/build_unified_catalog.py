import pandas as pd


INPUT_FILE = "data/raw/huggingface_datasets.csv"
OUTPUT_FILE = "data/processed/unified_catalog.csv"


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


df = pd.read_csv(INPUT_FILE)

# Ensure required unified columns exist
df["task"] = ""

# Clean text fields
df["title"] = df["title"].apply(clean_text)
df["description"] = df["description"].apply(clean_text)
df["tags"] = df["tags"].apply(clean_text)

# Create searchable text for recommender
df["search_text"] = (
    df["title"] + " " +
    df["description"] + " " +
    df["tags"]
).str.strip()

# Final unified schema
columns = [
    "dataset_id",
    "title",
    "description",
    "tags",
    "task",
    "source",
    "downloads",
    "likes",
    "url",
    "search_text"
]

df = df[columns]

df.to_csv(OUTPUT_FILE, index=False)

print("Unified catalog saved:", OUTPUT_FILE)
print("Shape:", df.shape)
print(df.head())