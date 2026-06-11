import pandas as pd
from huggingface_hub import list_datasets
from tqdm import tqdm


MAX_DATASETS = 10000

records = []

for i, ds in enumerate(tqdm(list_datasets(full=True), desc="Collecting Hugging Face datasets")):
    if i >= MAX_DATASETS:
        break

    records.append({
        "dataset_id": f"HF_{ds.id}",
        "title": ds.id,
        "description": getattr(ds, "description", None),
        "tags": ",".join(ds.tags) if ds.tags else None,
        "downloads": getattr(ds, "downloads", None),
        "likes": getattr(ds, "likes", None),
        "source": "huggingface",
        "url": f"https://huggingface.co/datasets/{ds.id}"
    })

df = pd.DataFrame(records)

df.to_csv("data/raw/huggingface_datasets.csv", index=False)

print("Saved:", df.shape)
print(df.head())