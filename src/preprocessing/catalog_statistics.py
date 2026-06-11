import pandas as pd

df = pd.read_csv("data/processed/unified_catalog.csv")

print("\nTotal datasets:")
print(len(df))

print("\nUnique sources:")
print(df["source"].value_counts())

print("\nTop 20 most downloaded datasets:")
print(
    df.sort_values("downloads", ascending=False)[
        ["title", "downloads", "likes"]
    ].head(20)
)

print("\nDatasets with descriptions:")
print(df["description"].str.len().gt(0).sum())

print("\nDatasets without descriptions:")
print(df["description"].str.len().eq(0).sum())

print("\nAverage downloads:")
print(df["downloads"].mean())

print("\nAverage likes:")
print(df["likes"].mean())