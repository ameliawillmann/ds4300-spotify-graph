"""
neo4j_connection.py
Connects and disconnects from the neo4j database.
"""

from neo4j import GraphDatabase

# ============================================================
# CONFIGURATION - Update these to match your Neo4j setup
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "CharGoes2NU2022!"


# ============================================================
# CONNECTION
# ============================================================

def connect():
    """
    Create and return a Neo4j driver instance.
    Verifies connectivity before returning.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"Connected to Neo4j at {NEO4J_URI}")
    return driver


def close(driver):
    """Close the Neo4j driver connection."""
    driver.close()
    print("Neo4j connection closed.")


def clear_database(driver):
    """Remove all existing nodes and relationships from the database."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    driver = connect()
    with driver.session() as session:
        result = session.run("RETURN 'Hello from Neo4j!' AS message")
        print(result.single()["message"])
    close(driver)