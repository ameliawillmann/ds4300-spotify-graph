"""
recommender.py
===============
Main entry point for the Spotify Song Recommendation Engine.

Ties together all modules:
    - neo4j_connection.py  → Database connection
    - data_preprocessing.py → Data loading, sampling, normalization, similarity
    - cypher_queries.py     → Graph construction and recommendation queries

Usage:
    python recommender.py

Make sure Neo4j is running before executing:
    neo4j start
"""

from neo4j_connection import connect, close, clear_database
from data_preprocessing import load_and_sample_data, normalize_features, compute_edges
from cypher_queries import (
    create_indexes,
    create_song_nodes,
    create_similarity_edges,
    get_graph_stats,
    get_recommendations,
    get_strokes_spektor_songs,
    get_degree_distribution
)


def main():
    # ========================================
    # STEP 1: Load and preprocess data
    # ========================================
    print("=" * 60)
    print("STEP 1: Loading and sampling Spotify data")
    print("=" * 60)
    df = load_and_sample_data()

    # ========================================
    # STEP 2: Normalize and compute similarity
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Normalizing features and computing similarity")
    print("=" * 60)
    df_norm = normalize_features(df)
    edges = compute_edges(df_norm)

    # ========================================
    # STEP 3: Connect to Neo4j and build graph
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: Building the Neo4j graph")
    print("=" * 60)
    driver = connect()
    clear_database(driver)
    create_indexes(driver)
    create_song_nodes(driver, df)
    create_similarity_edges(driver, df, edges)

    # ========================================
    # STEP 4: Graph stats and exploration
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: Graph statistics")
    print("=" * 60)
    get_graph_stats(driver)
    get_strokes_spektor_songs(driver)
    get_degree_distribution(driver)

    # ========================================
    # STEP 5: Generate recommendations
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 5: Generating recommendations")
    print("=" * 60)
    get_recommendations(driver, limit=5)

    # ========================================
    # Cleanup
    # ========================================
    close(driver)
    print("\nDone!")


if __name__ == "__main__":
    main()
