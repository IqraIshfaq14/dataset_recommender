import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/raw/huggingface_datasets.csv")

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate dataset IDs:")
print(df["dataset_id"].duplicated().sum())

print("\nDuplicate titles:")
print(df["title"].duplicated().sum())

df["description_words"] = (
    df["description"]
    .fillna("")
    .str.split()
    .str.len()
)

print("\nAverage description length:")
print(round(df["description_words"].mean(), 2))

print("\nMedian description length:")
print(df["description_words"].median())

def clean_tag(tag):
    tag = str(tag).strip()

    label_map = {
        "language:en": "English",
        "language:us": "United States",
        "region:us": "United States",
        "en": "English",
        "us": "United States",
        "size_categories:10K<n<100K": "10K–100K rows",
        "size_categories:1K<n<10K": "1K–10K rows",
        "size_categories:100K<n<1M": "100K–1M rows",
        "size_categories:1M<n<10M": "1M–10M rows",
        "10K<n<100K": "10K–100K rows",
        "1K<n<10K": "1K–10K rows",
        "100K<n<1M": "100K–1M rows",
        "1M<n<10M": "1M–10M rows",
    }

    if tag in label_map:
        return label_map[tag]

    for prefix in [
        "size_categories:",
        "library:",
        "modality:",
        "language:",
        "format:",
        "region:",
    ]:
        tag = tag.replace(prefix, "")

    return tag.replace("_", " ").title()

all_tags = (
    df["tags"]
    .fillna("")
    .str.split(",")
    .explode()
    .str.strip()
)

all_tags = all_tags[all_tags != ""]

cleaned_tags = all_tags.apply(clean_tag)

exclude_tags = {
    "Datasets",
    "Mlcroissant",
    "Pandas",
    "Polars",
    "Parquet",
    "Text",
}

cleaned_tags = cleaned_tags[~cleaned_tags.isin(exclude_tags)]

top_tags_short = cleaned_tags.value_counts().head(10)

print("\nTop 10 Tags:")
print(top_tags_short)

Path("reports/figures").mkdir(parents=True, exist_ok=True)

missing = df.isnull().sum()

plt.figure(figsize=(8, 5))
missing.plot(kind="bar")
plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Missing Count")
plt.tight_layout()
plt.savefig("reports/figures/missing_values.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
df["description_words"].plot(kind="hist", bins=50)
plt.title("Description Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Datasets")
plt.tight_layout()
plt.savefig("reports/figures/description_length_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
top_tags_short.sort_values().plot(kind="barh")
plt.title("Top 10 Dataset Tags")
plt.xlabel("Number of Datasets")
plt.ylabel("Tag")
plt.tight_layout()
plt.savefig("reports/figures/top_10_tags.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nCharts saved to reports/figures/")