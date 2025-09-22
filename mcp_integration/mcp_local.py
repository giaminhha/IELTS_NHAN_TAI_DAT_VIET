# mcp_server.py
from flask import Flask, request, jsonify
from neo4j import GraphDatabase
import time

app = Flask(__name__)

# --- Neo4j connection ---
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "NHAN_TAI_DAT_VIET_098"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# --- Cache dictionary ---
cache = {}
CACHE_EXPIRY = 10 * 60  # 10 minutes in seconds

def cache_get(key):
    entry = cache.get(key)
    if entry:
        value, timestamp = entry
        if time.time() - timestamp < CACHE_EXPIRY:
            return value
    return None

def cache_set(key, value):
    cache[key] = (value, time.time())

# --- Helper function to run a query ---
def run_query(query, params=None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]

# --- 1. Cached queries with expiry ---
def get_passage_rules():
    key = "passage_rules"
    cached = cache_get(key)
    if cached:
        return cached
    query = """
    MATCH (r:FormatRule)
    RETURN r.id AS id, r.description AS description
    ORDER BY r.id
    """
    result = run_query(query)
    cache_set(key, result)
    return result

def get_question_type_context(qtype_id):
    key = f"question_type_context:{qtype_id}"
    cached = cache_get(key)
    if cached:
        return cached
    query = """
    MATCH (q:QuestionType {id: $qtype_id})-[:HAS_RULE]->(r:QuestionTypeRule)
    RETURN r.id AS id, r.description AS description
    """
    result = run_query(query, {"qtype_id": qtype_id})
    cache_set(key, result)
    return result

def get_distractor_patterns():
    key = "distractor_patterns"
    cached = cache_get(key)
    if cached:
        return cached
    query = "MATCH (d:Distractor) RETURN d.id AS id, d.description AS description"
    result = run_query(query)
    cache_set(key, result)
    return result

def get_penmanship_rules():
    key = "penmanship_rules"
    cached = cache_get(key)
    if cached:
        return cached
    query = """
    MATCH (p:Penmanship)-[:HAS_SUBRULE]->(s:PenmanshipSubrule)
    RETURN p.id AS penmanship_id, p.description AS penmanship_desc,
           collect({id: s.id, description: s.description}) AS subrules
    """
    result = run_query(query)
    cache_set(key, result)
    return result
def get_passage_examples():
    key = "passage_examples"
    cached = cache_get(key)
    if cached:
        return cached
    query = """
    MATCH (p:Entity {type: "PassageExample"})
    RETURN p.id AS id, p.title AS title, p.passage AS passage
    ORDER BY p.id
    """
    result = run_query(query)
    cache_set(key, result)
    return result

def get_question_examples(qtype_id=None):
    if qtype_id:
        query = """
        MATCH (q:Entity {type: "QuestionExample"})-[:OF_TYPE]->(qt:Entity {id: $qtype_id})
        RETURN q.id AS id, q.question_text AS question_text, q.options AS options,
               q.answer AS answer, q.rationale AS rationale, qt.id AS qtype_id
        ORDER BY q.id
        """
        return run_query(query, {"qtype_id": qtype_id})
    else:
        query = """
        MATCH (q:Entity {type: "QuestionExample"})-[:OF_TYPE]->(qt:Entity)
        RETURN q.id AS id, q.question_text AS question_text, q.options AS options,
               q.answer AS answer, q.rationale AS rationale, qt.id AS qtype_id
        ORDER BY q.id
        """
        return run_query(query)
# --- 2. Flask routes ---
@app.route("/get_passage_rules", methods=["GET"])
def route_passage_rules():
    return jsonify(get_passage_rules())

@app.route("/get_question_type_context/<qtype_id>", methods=["GET"])
def route_question_type_context(qtype_id):
    return jsonify(get_question_type_context(qtype_id))

@app.route("/get_distractor_patterns", methods=["GET"])
def route_distractor_patterns():
    return jsonify(get_distractor_patterns())

@app.route("/get_penmanship_rules", methods=["GET"])
def route_penmanship_rules():
    return jsonify(get_penmanship_rules())
# --- 3. Example endpoints (optional style guidance) ---
@app.route("/get_passage_examples", methods=["GET"])
def route_passage_examples():
    return jsonify(get_passage_examples())

@app.route("/get_question_examples", methods=["GET"])
def route_question_examples_all():
    return jsonify(get_question_examples())

@app.route("/get_question_examples/<qtype_id>", methods=["GET"])
def route_question_examples_by_type(qtype_id):
    return jsonify(get_question_examples(qtype_id))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
