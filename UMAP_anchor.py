import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from umap import UMAP

import joblib


# ------------------------
# SETTINGS
# ------------------------

CSV_FILE = "anchor_test.csv"
N_SAMPLE = 10000

# ------------------------
# LOAD
# ------------------------

print("Loading data...")

df = pd.read_csv(CSV_FILE)

df = df.dropna(subset=["Title", "Abstract"])

# optional for first test
if len(df) > N_SAMPLE:
    df = df.sample(
        N_SAMPLE,
        random_state=42
    )

print(f"Papers: {len(df):,}")

# ------------------------
# TEXT FOR EMBEDDING
# ------------------------

texts = (
    df["Title"].fillna("")
    + ". "
    + df["Abstract"].fillna("")
).tolist()

# ------------------------
# EMBEDDINGS
# ------------------------

print("Loading SPECTER model...")

model = SentenceTransformer(
    "allenai-specter"
)

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
)


np.save(
    "anchor_embeddings.npy",
    embeddings
)


# ------------------------
# UMAP
# ------------------------

print("Running UMAP...")

umap_model = UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)

coords = umap_model.fit_transform(embeddings)

joblib.dump(
    umap_model,
    "umap_model.pkl"
)


df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# ------------------------
# SAVE
# ------------------------

outfile = "paper_map.parquet"

df.to_parquet(outfile)

print(f"Saved {outfile}")