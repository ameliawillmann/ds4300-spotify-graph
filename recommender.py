
"""
Amelia Willmann, Katie Malan, Charlotte Thunen


"""
from neo4j_connection import connect, close, clear_database
from data_preprocessing import load_and_sample_data, normalize_features, compute_edges
from cypher_queries import build_graph, explore_graph, get_recommendations


def prompt_liked_artists():
    """Prompt the user to enter 2 liked artists."""
    print("\n" + "=" * 60)
    print("MUSIC RECOMMENDATION SYSTEM")
    print("=" * 60)
    artist1 = input("Enter your first liked artist: ").strip().title()
    artist2 = input("Enter your second liked artist: ").strip().title()
    return [artist1, artist2]


def main():
    liked_artists = prompt_liked_artists()
    print(f"\nGenerating recommendations based on: {liked_artists[0]} & {liked_artists[1]}")

    df = load_and_sample_data(liked_artists=liked_artists)
    df_norm = normalize_features(df)
    edges = compute_edges(df_norm)

    driver = connect()
    clear_database(driver)
    build_graph(driver, df, edges)
    explore_graph(driver, liked_artists=liked_artists)
    get_recommendations(driver, liked_artists=liked_artists, limit=5)
    close(driver)


if __name__ == "__main__":
    main()