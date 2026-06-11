import pandas as pd

df = pd.read_csv("data/processed/unified_catalog.csv")

REMOVE_PREFIXES = [
    "library:",
    "format:",
    "region:"
]


def clean_tags(tag_string):
    if pd.isna(tag_string):
        return ""

    tags = tag_string.split(",")

    cleaned = []

    for tag in tags:
        tag = tag.strip()

        keep = True

        for prefix in REMOVE_PREFIXES:
            if tag.startswith(prefix):
                keep = False
                break

        if keep:
            cleaned.append(tag)

    return ",".join(cleaned)


df["clean_tags"] = df["tags"].apply(clean_tags)

print("\nBefore:")
print(df["tags"].head(3))

print("\nAfter:")
print(df["clean_tags"].head(3))

df.to_csv(
    "data/processed/unified_catalog_clean_tags.csv",
    index=False
)

print("\nSaved:")
print("data/processed/unified_catalog_clean_tags.csv")