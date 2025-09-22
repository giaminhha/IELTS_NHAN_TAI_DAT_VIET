#kg_queries.py
from neo4j import GraphDatabase

# Configure Neo4j connection (edit as needed)
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "NHAN_TAI_DAT_VIET_098"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def query_kg(node, relations=None, depth=2):
    """
    Query KG with up to 'depth' hops along allowed relations.
    Returns list of dicts: {"path": [...], "target": "...", "desc": "..."}
    """
    with driver.session() as session:
        if relations:
            query = f"""
            MATCH (n {{id: $node}})-[r*1..{depth}]->(m)
            WHERE ALL(rel IN r WHERE type(rel) IN $relations)
            RETURN [rel IN r | type(rel)] AS path, m.id AS target, m.description AS desc
            """
            result = session.run(query, node=node, relations=relations)
        else:
            query = f"""
            MATCH (n {{id: $node}})-[r*1..{depth}]->(m)
            RETURN [rel IN r | type(rel)] AS path, m.id AS target, m.description AS desc
            """
            result = session.run(query, node=node)

        return [dict(r) for r in result]
