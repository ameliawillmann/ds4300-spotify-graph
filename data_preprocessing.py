"""
data_preprocessing.py
Handles loading, sampling, normalizing the Spotify data,
and computing pairwise similarity between songs.

SAMPLING STRATEGY:
- Include ALL songs by The Strokes and Regina Spektor
- Randomly sample remaining songs across genres for diversity
- Remove duplicate track_ids

SIMILARITY METRIC:
- Euclidean distance on min-max normalized musical features
- Features used: danceability, energy, speechiness,
  acousticness, instrumentalness, liveness, valence, tempo
"""

import pandas as pd
import numpy as np
from itertools import combinations

CSV_PATH = "spotify.csv"
SAMPLE_SIZE = 6000
SIMILARITY_THRESHOLD = 0.30

FEATURES = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]


# ============================================================
# DATA LOADING AND SAMPLING
# ============================================================

def load_and_sample_data(csv_path=CSV_PATH, sample_size=SAMPLE_SIZE):
    """
    Load the Spotify CSV and create a sample that:
    - Includes ALL songs by The Strokes and Regina Spektor
    - Randomly samples remaining songs for diversity
    - Removes duplicate track_ids (keeps first occurrence)

    Returns:
        pd.DataFrame: Sampled dataset
    """
    print("Loading Spotify data...")
    df = pd.read_csv(csv_path)
    print(f"  Full dataset: {len(df):,} songs")

    df = df.drop_duplicates(subset='track_id', keep='first')
    print(f"  After deduplication: {len(df):,} songs")

    strokes_mask = df['artists'].str.contains('The Strokes', case=False, na=False)
    spektor_mask = df['artists'].str.contains('Regina Spektor', case=False, na=False)
    must_include = df[strokes_mask | spektor_mask]
    remaining = df[~(strokes_mask | spektor_mask)]

    print(f"  The Strokes songs: {strokes_mask.sum()}")
    print(f"  Regina Spektor songs: {spektor_mask.sum()}")

    n_random = sample_size - len(must_include)
    random_sample = remaining.sample(n=min(n_random, len(remaining)), random_state=42)

    sample = pd.concat([must_include, random_sample]).reset_index(drop=True)
    print(f"  Final sample size: {len(sample):,} songs")
    return sample


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(df, features=FEATURES):
    """
    Min-max normalize musical features to [0, 1].
    Ensures all features contribute equally to distance.

    Returns:
        pd.DataFrame: Copy of df with normalized feature columns
    """
    print("Normalizing features...")
    df_norm = df.copy()
    for feat in features:
        col = df_norm[feat].astype(float)
        min_val, max_val = col.min(), col.max()
        df_norm[feat] = (col - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    print(f"  Normalized {len(features)} features: {features}")
    return df_norm


# ============================================================
# SIMILARITY COMPUTATION
# ============================================================

def compute_edges(df, features=FEATURES, threshold=SIMILARITY_THRESHOLD):
    """
    Compute Euclidean distance between all pairs of songs.
    Only keeps pairs whose distance is below the threshold.

    Returns:
        list of tuples: (index_i, index_j, distance)
    """
    print(f"Computing pairwise Euclidean distances (threshold={threshold})...")
    feature_matrix = df[features].values.astype(float)

    edges = []
    n = len(feature_matrix)
    total_pairs = n * (n - 1) // 2

    for count, (i, j) in enumerate(combinations(range(n), 2)):
        dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
        if dist <= threshold:
            edges.append((i, j, round(float(dist), 6)))

        if (count + 1) % 500_000 == 0:
            pct = (count + 1) / total_pairs * 100
            print(f"  {pct:.1f}% done | {count + 1:,}/{total_pairs:,} pairs | Edges: {len(edges):,}")

    print(f"  Complete! Total edges: {len(edges):,}")
    return edges
