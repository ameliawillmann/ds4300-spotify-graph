
"""
recommender.py
==============
Main entry point for the Spotify Song Recommendation Engine.

Usage:
    python recommender.py

Make sure Neo4j is running before executing:
    neo4j start
"""

from neo4j_connection import connect, close, clear_database

from data_preprocessing import load_and_sample_data, normalize_features, compute_edges
from cypher_queries import build_graph, explore_graph, get_recommendations


def main():
    df = load_and_sample_data()
    df_norm = normalize_features(df)
    edges = compute_edges(df_norm)

    driver = connect()
    clear_database(driver)
    build_graph(driver, df, edges)
    explore_graph(driver)
    get_recommendations(driver, limit=5)
    close(driver)


if __name__ == "__main__":
    main()