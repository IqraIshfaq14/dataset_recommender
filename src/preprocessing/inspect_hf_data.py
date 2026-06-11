import pandas as pd

df = pd.read_csv("data/raw/huggingface_datasets.csv")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSample rows:")
print(df.head())

print("\nTop downloaded datasets:")
print(
    df.sort_values("downloads", ascending=False)[
        ["title", "downloads", "likes"]
    ].head(10)
)
print("\nDescription coverage:")
print(
    f"{100 * (1 - df['description'].isna().sum() / len(df)):.2f}%"
)
