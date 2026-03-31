"""
data_preprocessing.py
======================
Handles loading, sampling, normalizing the Spotify data,
and computing pairwise similarity between songs.

SAMPLING STRATEGY:
- Include ALL songs by The Strokes and Regina Spektor
- Randomly sample remaining songs across genres for diversity
- Remove duplicate track_ids

SIMILARITY METRIC:
- Euclidean distance on min-max normalized musical features
- Features used: danceability, energy, loudness, speechiness,
  acousticness, instrumentalness, liveness, valence, tempo
"""

import pandas as pd
import numpy as np
from itertools import combinations


# ============================================================
# CONFIGURATION
# ============================================================

# Path to the Spotify CSV file - UPDATE THIS
CSV_PATH = "/Users/charlottethunen/Documents/Northeastern/4th_yr/DS4300/HW5/ds4300-spotify-graph/spotify.csv"

SAMPLE_SIZE = 2000           # Total number of songs to sample
SIMILARITY_THRESHOLD = 0.15  # Max Euclidean distance for creating an edge

# Musical features used for computing similarity
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]


# ============================================================
# DATA LOADING AND SAMPLING
# ============================================================

def load_and_sample_data(csv_path=CSV_PATH, sample_size=SAMPLE_SIZE):
    """
    Load the Spotify CSV and create a sample that:
    - Includes ALL songs by The Strokes and Regina Spektor
    - Randomly samples remaining songs across genres for diversity
    - Removes duplicate track_ids (keeps first occurrence)

    Returns:
        pd.DataFrame: Sampled dataset
    """
    print("Loading Spotify data...")
    df = pd.read_csv(csv_path)
    print(f"  Full dataset: {len(df):,} songs")

    # Remove duplicates by track_id (some songs appear in multiple genres)
    df = df.drop_duplicates(subset='track_id', keep='first')
    print(f"  After deduplication: {len(df):,} songs")

    # Separate must-include artists
    strokes_mask = df['artists'].str.contains('The Strokes', case=False, na=False)
    spektor_mask = df['artists'].str.contains('Regina Spektor', case=False, na=False)
    must_include = df[strokes_mask | spektor_mask]
    remaining = df[~(strokes_mask | spektor_mask)]

    print(f"  The Strokes songs: {strokes_mask.sum()}")
    print(f"  Regina Spektor songs: {spektor_mask.sum()}")

    # Randomly sample the rest
    n_random = sample_size - len(must_include)
    random_sample = remaining.sample(n=min(n_random, len(remaining)), random_state=42)

    # Combine into final sample
    sample = pd.concat([must_include, random_sample]).reset_index(drop=True)
    print(f"  Final sample size: {len(sample):,} songs")

    return sample


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(df, features=FEATURES):
    """
    Min-max normalize musical features to [0, 1] range.
    This ensures all features contribute equally to distance.

    Args:
        df: DataFrame with raw feature values
        features: List of column names to normalize

    Returns:
        pd.DataFrame: Copy of df with normalized feature columns
    """
    print("Normalizing features...")
    df_norm = df.copy()
    for feat in features:
        col = df_norm[feat].astype(float)
        min_val, max_val = col.min(), col.max()
        if max_val - min_val > 0:
            df_norm[feat] = (col - min_val) / (max_val - min_val)
        else:
            df_norm[feat] = 0.0
    print(f"  Normalized {len(features)} features: {features}")
    return df_norm


# ============================================================
# SIMILARITY COMPUTATION
# ============================================================

def compute_edges(df, features=FEATURES, threshold=SIMILARITY_THRESHOLD):
    """
    Compute Euclidean distance between all pairs of songs.
    Only keep pairs whose distance is below the threshold.

    Args:
        df: DataFrame with normalized features
        features: List of feature columns to use
        threshold: Maximum distance to create an edge

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

        # Progress update every 500K pairs
        if (count + 1) % 500_000 == 0:
            pct = (count + 1) / total_pairs * 100
            print(f"  {pct:.1f}% done | {count + 1:,}/{total_pairs:,} pairs | Edges: {len(edges):,}")

    print(f"  Complete! Total edges: {len(edges):,}")
    return edges


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    # Test the full preprocessing pipeline
    df = load_and_sample_data()
    df_norm = normalize_features(df)
    edges = compute_edges(df_norm)

    print(f"\nSample preview:")
    print(df[['track_name', 'artists', 'track_genre']].head(10))
    print(f"\nEdge sample (first 5): {edges[:5]}")
