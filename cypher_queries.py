"""
cypher_queries.py
==================
Contains all Cypher queries for building and querying the Neo4j graph.

GRAPH DATA MODEL:
-----------------
Nodes:
    (:Song) - Each song is a node with properties:
        - track_id (string, unique identifier)
        - track_name, artists, album_name (string)
        - popularity (integer)
        - danceability, energy, loudness, speechiness (float)
        - acousticness, instrumentalness, liveness, valence, tempo (float)
        - track_genre (string)

Relationships:
    (:Song)-[:SIMILAR_TO {distance: float}]->(:Song)
        - Bidirectional edges connecting musically similar songs
        - distance: Euclidean distance between normalized feature vectors
        - Only created when distance <= threshold (default 0.15)
"""


# ============================================================
# INDEX CREATION
# ============================================================

def create_indexes(driver):
    """Create indexes on Song nodes for faster lookups."""
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Song) ON (s.track_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Song) ON (s.artists)")
    print("Indexes created on track_id and artists.")


# ============================================================
# NODE CREATION
# ============================================================

def create_song_nodes(driver, df):
    """
    Create Song nodes in Neo4j from the sampled DataFrame.
    Uses MERGE on track_id to avoid duplicates.

    Args:
        driver: Neo4j driver instance
        df: DataFrame containing song data
    """
    print(f"Creating {len(df):,} Song nodes...")
    with driver.session() as session:
        for idx, row in df.iterrows():
            session.run("""
                MERGE (s:Song {track_id: $track_id})
                SET s.track_name = $track_name,
                    s.artists = $artists,
                    s.album_name = $album_name,
                    s.popularity = $popularity,
                    s.danceability = $danceability,
                    s.energy = $energy,
                    s.loudness = $loudness,
                    s.speechiness = $speechiness,
                    s.acousticness = $acousticness,
                    s.instrumentalness = $instrumentalness,
                    s.liveness = $liveness,
                    s.valence = $valence,
                    s.tempo = $tempo,
                    s.track_genre = $track_genre
            """, {
                'track_id': str(row['track_id']),
                'track_name': str(row['track_name']),
                'artists': str(row['artists']),
                'album_name': str(row['album_name']),
                'popularity': int(row['popularity']),
                'danceability': float(row['danceability']),
                'energy': float(row['energy']),
                'loudness': float(row['loudness']),
                'speechiness': float(row['speechiness']),
                'acousticness': float(row['acousticness']),
                'instrumentalness': float(row['instrumentalness']),
                'liveness': float(row['liveness']),
                'valence': float(row['valence']),
                'tempo': float(row['tempo']),
                'track_genre': str(row['track_genre'])
            })

            if (idx + 1) % 500 == 0:
                print(f"  Created {idx + 1:,}/{len(df):,} nodes...")

    print(f"  All {len(df):,} Song nodes created.")


# ============================================================
# EDGE CREATION
# ============================================================

def create_similarity_edges(driver, df, edges):
    """
    Create SIMILAR_TO relationships between similar songs.
    Each edge stores the Euclidean distance as a property.
    Edges are bidirectional.

    Args:
        driver: Neo4j driver instance
        df: DataFrame (used to look up track_ids by index)
        edges: List of (index_i, index_j, distance) tuples
    """
    print(f"Creating {len(edges):,} SIMILAR_TO edges...")
    with driver.session() as session:
        for count, (i, j, dist) in enumerate(edges):
            track_id_i = str(df.iloc[i]['track_id'])
            track_id_j = str(df.iloc[j]['track_id'])
            session.run("""
                MATCH (a:Song {track_id: $id1}), (b:Song {track_id: $id2})
                MERGE (a)-[:SIMILAR_TO {distance: $dist}]->(b)
                MERGE (b)-[:SIMILAR_TO {distance: $dist}]->(a)
            """, {'id1': track_id_i, 'id2': track_id_j, 'dist': dist})

            if (count + 1) % 5000 == 0:
                print(f"  Created {count + 1:,}/{len(edges):,} edges...")

    print(f"  All {len(edges):,} SIMILAR_TO edges created.")


# ============================================================
# GRAPH STATISTICS
# ============================================================

def get_graph_stats(driver):
    """Print and return total node and edge counts."""
    with driver.session() as session:
        nodes = session.run("MATCH (n:Song) RETURN COUNT(n) AS count").single()["count"]
        edges = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN COUNT(r) AS count").single()["count"]

    print(f"\nGraph Statistics:")
    print(f"  Total Song nodes: {nodes:,}")
    print(f"  Total SIMILAR_TO edges: {edges:,}")
    return nodes, edges


# ============================================================
# RECOMMENDATION QUERY
# ============================================================

def get_recommendations(driver, limit=5):
    """
    Generate song recommendations for Prof. Rachlin.

    Algorithm:
    - Start from all songs by The Strokes and Regina Spektor
    - Traverse SIMILAR_TO edges to find neighboring songs
    - Rank by: number of connections to liked songs (more = better),
      then by average distance (lower = more similar)
    - Exclude songs by The Strokes or Regina Spektor

    Args:
        driver: Neo4j driver instance
        limit: Number of recommendations to return

    Returns:
        list of records with Song, Artist, Album, Genre, connections, avg_distance
    """
    query = """
        MATCH (liked:Song)
        WHERE liked.artists CONTAINS 'The Strokes'
           OR liked.artists CONTAINS 'Regina Spektor'

        MATCH (liked)-[r:SIMILAR_TO]->(rec:Song)

        WHERE NOT rec.artists CONTAINS 'The Strokes'
          AND NOT rec.artists CONTAINS 'Regina Spektor'

        RETURN rec.track_name AS Song,
               rec.artists AS Artist,
               rec.album_name AS Album,
               rec.track_genre AS Genre,
               COUNT(r) AS connections,
               ROUND(AVG(r.distance), 4) AS avg_distance
        ORDER BY connections DESC, avg_distance ASC
        LIMIT $limit
    """

    print("\n" + "=" * 60)
    print(f"TOP {limit} SONG RECOMMENDATIONS FOR PROF. RACHLIN")
    print("=" * 60)

    with driver.session() as session:
        result = session.run(query, {"limit": limit})
        records = list(result)

        if not records:
            print("No recommendations found. Try adjusting the similarity threshold.")
            return []

        for i, record in enumerate(records, 1):
            print(f"\n  {i}. \"{record['Song']}\"")
            print(f"     Artist: {record['Artist']}")
            print(f"     Album:  {record['Album']}")
            print(f"     Genre:  {record['Genre']}")
            print(f"     Connections to liked songs: {record['connections']}")
            print(f"     Avg similarity distance:    {record['avg_distance']}")

    print("\n" + "=" * 60)
    return records


# ============================================================
# ADDITIONAL USEFUL QUERIES
# ============================================================

def get_strokes_spektor_songs(driver):
    """List all Strokes and Regina Spektor songs in the graph."""
    query = """
        MATCH (s:Song)
        WHERE s.artists CONTAINS 'The Strokes'
           OR s.artists CONTAINS 'Regina Spektor'
        RETURN s.track_name AS Song, s.artists AS Artist,
               s.album_name AS Album, s.track_genre AS Genre
        ORDER BY s.artists, s.album_name, s.track_name
    """
    with driver.session() as session:
        result = session.run(query)
        records = list(result)

    print(f"\nStrokes & Spektor songs in graph: {len(records)}")
    for r in records:
        print(f"  {r['Artist']} - \"{r['Song']}\" ({r['Album']})")
    return records


def get_neighbors_of_song(driver, track_name):
    """Find all songs similar to a specific song."""
    query = """
        MATCH (s:Song {track_name: $name})-[r:SIMILAR_TO]->(neighbor:Song)
        RETURN neighbor.track_name AS Song, neighbor.artists AS Artist,
               neighbor.track_genre AS Genre, r.distance AS distance
        ORDER BY r.distance ASC
        LIMIT 10
    """
    with driver.session() as session:
        result = session.run(query, {"name": track_name})
        records = list(result)

    print(f"\nTop 10 songs similar to \"{track_name}\":")
    for r in records:
        print(f"  {r['Artist']} - \"{r['Song']}\" ({r['Genre']}) | dist: {r['distance']:.4f}")
    return records


def get_degree_distribution(driver):
    """Show how many connections each song has (degree distribution)."""
    query = """
        MATCH (s:Song)-[r:SIMILAR_TO]->()
        WITH s, COUNT(r) AS degree
        RETURN degree, COUNT(s) AS num_songs
        ORDER BY degree DESC
    """
    with driver.session() as session:
        result = session.run(query)
        records = list(result)

    print(f"\nDegree Distribution:")
    for r in records[:15]:
        print(f"  Degree {r['degree']}: {r['num_songs']} songs")
    return records
