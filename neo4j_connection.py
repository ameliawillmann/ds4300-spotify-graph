"""
Amelia Willmann, Katie Malan, Charlotte Thunen
"""
from neo4j import GraphDatabase
import os

"""
Set up environmental variables in terminal:
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_actual_password"

Check they are set up:
echo $NEO4J_URI
echo $NEO4J_PASSWORD
"""
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def connect():
    """Create and return Neo4j driver & verifies connectivity"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"Connected to Neo4j at {NEO4J_URI}")
    return driver

def clear_database(driver):
    """Remove all nodes and relationships in batches to avoid memory issues."""
    with driver.session() as session:
        # Delete relationships in batches
        while True:
            result = session.run("""
                MATCH ()-[r]->()
                WITH r LIMIT 10000
                DELETE r
                RETURN COUNT(r) AS deleted
            """)
            deleted = result.single()["deleted"]
            if deleted == 0:
                break
            print(f"  Deleted {deleted:,} relationships...")

        # Delete nodes in batches
        while True:
            result = session.run("""
                MATCH (n)
                WITH n LIMIT 10000
                DELETE n
                RETURN COUNT(n) AS deleted
            """)
            deleted = result.single()["deleted"]
            if deleted == 0:
                break
            print(f"  Deleted {deleted:,} nodes...")

    print("Database cleared.")

def close(driver):
    """Close the Neo4j driver connection"""
    driver.close()
    print("Neo4j connection closed.")