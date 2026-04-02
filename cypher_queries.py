BATCH_SIZE = 500


def create_indexes(driver):
    """Create indexes on Song nodes for faster lookups."""
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Song) ON (s.track_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Song) ON (s.artists)")
    print("Indexes created on track_id and artists.")


def create_song_nodes(driver, df):
    """Batch-insert Song nodes using MERGE on track_id to avoid duplicates."""
    print(f"Creating {len(df):,} Song nodes...")
    records = df[[
        'track_id', 'track_name', 'artists', 'album_name', 'popularity',
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre'
    ]].astype({
        'track_id': str, 'track_name': str, 'artists': str, 'album_name': str,
        'popularity': int, 'danceability': float, 'energy': float, 'loudness': float,
        'speechiness': float, 'acousticness': float, 'instrumentalness': float,
        'liveness': float, 'valence': float, 'tempo': float, 'track_genre': str
    }).to_dict('records')

    query = """
        UNWIND $batch AS row
        MERGE (s:Song {track_id: row.track_id})
        SET s.track_name       = row.track_name,
            s.artists          = row.artists,
            s.album_name       = row.album_name,
            s.popularity       = row.popularity,
            s.danceability     = row.danceability,
            s.energy           = row.energy,
            s.loudness         = row.loudness,
            s.speechiness      = row.speechiness,
            s.acousticness     = row.acousticness,
            s.instrumentalness = row.instrumentalness,
            s.liveness         = row.liveness,
            s.valence          = row.valence,
            s.tempo            = row.tempo,
            s.track_genre      = row.track_genre
    """
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            session.run(query, {"batch": records[i:i + BATCH_SIZE]})
            print(f"  {min(i + BATCH_SIZE, len(records)):,}/{len(records):,} nodes...")
    print(f"  All {len(records):,} Song nodes created.")


def create_similarity_edges(driver, df, edges):
    """Batch-insert bidirectional SIMILAR_TO edges with distance property."""
    print(f"Creating {len(edges):,} SIMILAR_TO edges...")
    track_ids = df['track_id'].astype(str).tolist()
    edge_records = [
        {"id1": track_ids[i], "id2": track_ids[j], "dist": dist}
        for i, j, dist in edges
    ]
    query = """
        UNWIND $batch AS row
        MATCH (a:Song {track_id: row.id1}), (b:Song {track_id: row.id2})
        MERGE (a)-[:SIMILAR_TO {distance: row.dist}]->(b)
        MERGE (b)-[:SIMILAR_TO {distance: row.dist}]->(a)
    """
    with driver.session() as session:
        for i in range(0, len(edge_records), BATCH_SIZE):
            session.run(query, {"batch": edge_records[i:i + BATCH_SIZE]})
            if (i + BATCH_SIZE) % 5000 == 0 or i + BATCH_SIZE >= len(edge_records):
                print(f"  {min(i + BATCH_SIZE, len(edge_records)):,}/{len(edge_records):,} edges...")
    print(f"  All {len(edge_records):,} SIMILAR_TO edges created.")


def get_graph_stats(driver):
    """Print and return total node and edge counts."""
    with driver.session() as session:
        nodes = session.run("MATCH (n:Song) RETURN COUNT(n) AS count").single()["count"]
        edges = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN COUNT(r) AS count").single()["count"]
    print(f"\nGraph Statistics:")
    print(f"  Total Song nodes: {nodes:,}")
    print(f"  Total SIMILAR_TO edges: {edges:,}")
    return nodes, edges


def get_recommendations(driver, liked_artists, limit=5):
    """
    Recommend songs similar to liked_artists, ranked by number of
    connections to liked songs (desc) then avg distance (asc).
    """

    query = """
        MATCH (liked:Song)
        WHERE ANY(artist IN $artists WHERE liked.artists CONTAINS artist)
        MATCH (liked)-[r:SIMILAR_TO]->(rec:Song)
        WHERE NOT ANY(artist IN $artists WHERE rec.artists CONTAINS artist)
        RETURN rec.track_name AS Song,
               rec.artists    AS Artist,
               rec.album_name AS Album,
               rec.track_genre AS Genre,
               COUNT(r) AS connections,
               ROUND(AVG(r.distance), 4) AS avg_distance
        ORDER BY connections DESC, avg_distance ASC
        LIMIT $limit
    """
    print("\n" + "=" * 60)
    print(f"TOP {limit} SONG RECOMMENDATIONS")
    print(f"Based on: {' & '.join(liked_artists)}")
    print("=" * 60)

    with driver.session() as session:
        records = list(session.run(query, {"limit": limit, "artists": liked_artists}))

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


def get_liked_songs(driver, liked_artists):
    """List all liked artist songs in the graph."""

    query = """
        MATCH (s:Song)
        WHERE ANY(artist IN $artists WHERE s.artists CONTAINS artist)
        RETURN s.track_name AS Song, s.artists AS Artist,
               s.album_name AS Album, s.track_genre AS Genre
        ORDER BY s.artists, s.album_name, s.track_name
    """
    with driver.session() as session:
        records = list(session.run(query, {"artists": liked_artists}))
    print(f"\nLiked artist songs in graph: {len(records)}")
    for r in records:
        print(f"  {r['Artist']} - \"{r['Song']}\" ({r['Album']})")
    return records


def get_neighbors_of_song(driver, track_name):
    """Find all similar songs to a given track, ordered by distance."""
    query = """
        MATCH (s:Song {track_name: $name})-[r:SIMILAR_TO]->(neighbor:Song)
        RETURN neighbor.track_name AS Song, neighbor.artists AS Artist,
               neighbor.track_genre AS Genre, r.distance AS distance
        ORDER BY r.distance ASC
    """
    with driver.session() as session:
        records = list(session.run(query, {"name": track_name}))
    print(f"\nSongs similar to \"{track_name}\" ({len(records)} total):")
    for r in records:
        print(f"  {r['Artist']} - \"{r['Song']}\" ({r['Genre']}) | dist: {r['distance']:.4f}")
    return records


def get_degree_distribution(driver):
    """Print degree distribution (number of SIMILAR_TO connections per song)."""
    query = """
        MATCH (s:Song)-[r:SIMILAR_TO]->()
        WITH s, COUNT(r) AS degree
        RETURN degree, COUNT(s) AS num_songs
        ORDER BY degree DESC
    """
    with driver.session() as session:
        records = list(session.run(query))
    print(f"\nDegree Distribution:")
    for r in records[:15]:
        print(f"  Degree {r['degree']}: {r['num_songs']} songs")
    return records


def build_graph(driver, df, edges):
    """Populate the database with song nodes and similarity edges."""
    create_indexes(driver)
    create_song_nodes(driver, df)
    create_similarity_edges(driver, df, edges)


def explore_graph(driver, liked_artists):
    """Print graph stats and artist/degree breakdowns."""
    get_graph_stats(driver)
    get_liked_songs(driver, liked_artists)
    get_degree_distribution(driver)