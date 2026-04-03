"""
Amelia Willmann, Katie Malan, Charlotte Thunen
Loads, samples, and normalizes Spotify dataset.
Computes pairwise weighted Euclidean similarity between songs.

SAMPLING STRATEGY:
- Include ALL songs by the two user-specified liked artists
- Randomly sample remaining songs across genres for diversity
- Remove duplicate track_ids

SIMILARITY METRIC:
- Weighted Euclidean distance on min-max normalized audio features
- Features are normalized to [0, 1] so no single feature dominates due to scale differences
- Weights are then applied to emphasize musically important features

FEATURES USED:
    danceability, energy, speechiness, acousticness,
    instrumentalness, liveness, valence, tempo

FEATURES INTENTIONALLY EXCLUDED:
    - genre: build a song recommender that will recommend songs that are audibly similar but will
            push the user to explore outside their usual genres
    - popularity: could bias results toward mainstream songs rather than audio similarity
    - loudness: not super indictive of similar songs
    - duration_ms: not meaningful for similarity
"""

import pandas as pd
import numpy as np
from itertools import combinations

CSV_PATH = "spotify.csv"
SAMPLE_SIZE = 5000
SIMILARITY_THRESHOLD = 0.25

FEATURES = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

FEATURE_WEIGHTS = {
    'danceability': 1.5,
    'energy': 1.5,
    'speechiness': 0.5,
    'acousticness': 1.0,
    'instrumentalness': 1.0,
    'liveness': 0.5, 
    'valence': 1,
    'tempo': 0.5
}

def load_and_sample_data(csv_path=CSV_PATH, sample_size=SAMPLE_SIZE, liked_artists=None):
    """
    Load the Spotify data. Create sample that:
    - Includes all songs by the two liked artists
    - Randomly samples remaining songs across genres
    - Removes duplicate track_ids
    """
    if liked_artists is None:
        # our chosen fallback artists
        liked_artists = ['Adele', 'Bob Dylan']

    print("Loading Spotify data...")
    df = pd.read_csv(csv_path)
    print(f"  Full dataset: {len(df):,} songs")

    df = df.drop_duplicates(subset='track_id', keep='first')
    print(f"  After dropping duplicates: {len(df):,} songs")

    masks = [df['artists'].str.contains(a, case=False, na=False) for a in liked_artists]
    must_include_mask = masks[0] | masks[1]
    must_include = df[must_include_mask]
    remaining = df[~must_include_mask]

    for artist, mask in zip(liked_artists, masks):
        print(f"  {artist} songs: {mask.sum()}")

    n_random = sample_size - len(must_include)
    random_sample = remaining.sample(n=min(n_random, len(remaining)), random_state=42)

    sample = pd.concat([must_include, random_sample]).reset_index(drop=True)
    print(f"  Final sample size: {len(sample):,} songs")
    return sample

def normalize_features(df, features=FEATURES):
    """
    Min-max normalize musical features to [0, 1].
    Ensures all features contribute equally to distance.
    """
    print("Normalizing features...")
    df_norm = df.copy()
    for feat in features:
        col = df_norm[feat].astype(float)
        min_val, max_val = col.min(), col.max()
        df_norm[feat] = (col - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    return df_norm

def apply_weights(feature_matrix, features=FEATURES, weights=FEATURE_WEIGHTS):
    """
    Scale each feature column by its weight.
    Weighted Euclidean distance = distance on this scaled matrix.
    """
    weight_vector = np.array([weights[f] for f in features])
    return feature_matrix * weight_vector

def compute_edges(df, features=FEATURES, threshold=SIMILARITY_THRESHOLD, weights=FEATURE_WEIGHTS):
    """
    Compute Euclidean distance between all pairs of songs.
    Only keeps pairs whose distance is below the threshold.
    """
    print(f"Computing pairwise Euclidean distances (threshold={threshold})...")
    feature_matrix = df[features].values.astype(float)
    weighted_matrix = apply_weights(feature_matrix, features, weights)

    edges = []
    n = len(weighted_matrix)
    total_pairs = n * (n - 1) // 2

    for count, (i, j) in enumerate(combinations(range(n), 2)):
        dist = np.linalg.norm(weighted_matrix[i] - weighted_matrix[j])
        if dist <= threshold:
            edges.append((i, j, round(float(dist), 6)))

        if (count + 1) % 500_000 == 0:
            pct = (count + 1) / total_pairs * 100
            print(f"  {pct:.1f}% done | {count + 1:,}/{total_pairs:,} pairs | Edges: {len(edges):,}")

    print(f"  Complete! Total edges: {len(edges):,}")
    return edges