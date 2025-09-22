# to_kg.py
import json
from neo4j import GraphDatabase

# ---- 1. Connect to Neo4j ----
URI = "bolt://localhost:7687"   # or your Aura URI if cloud
USER = "neo4j"                  # default user in Neo4j Desktop
PASSWORD = "NHAN_TAI_DAT_VIET_098" # the password you set

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# ---- 2. Load JSON ----
with open("lists_with_types.json", "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]
links = data["links"]

# ---- 3. Batch Node Insert ----
# ---- 3. Batch Node Insert ----
def create_nodes_batch(tx, rows):
    for row in rows:
        label = row.get("type", "Generic")   # fallback if type is missing
        props = {k: v for k, v in row.items() if k != "id" and k != "type"}
        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $props
        """
        tx.run(query, id=row["id"], props=props)


# ---- 4. Batch Relationship Insert ----
def create_rels_batch(tx, rows, rel_type):
    query = f"""
    UNWIND $rows AS row
    MATCH (a {{id: row.from}}), (b {{id: row.to}})
    MERGE (a)-[r:{rel_type}]->(b)
    """
    tx.run(query, rows=rows)

def collect_nodes():
    all_nodes = []

    categories = {
        "format_rules": "FormatRule",
        "writing_styles": "WritingStyle",
        "skills": "Skill",
        "distractors": "Distractor",
        "penmanship": "Penmanship",
        "question_types": "QuestionType",
        "question_type_rules": "QuestionTypeRule",
        "answer_behaviours": "AnswerBehaviour",
        "example_patterns": "ExamplePattern"
    }

    for n in nodes:
        # --- parent node ---
        all_nodes.append({
            "id": n["id"],
            "type": n["type"],
            "description": n.get("description", "")
        })

        # --- passage examples ---
        if "passage_examples" in n:
            for p in n["passage_examples"]:
                all_nodes.append({
                    "id": f"Passage_{p['id']}",
                    "type": "PassageExample",
                    "title": p.get("title", ""),
                    "passage": p.get("passage", "")
                })

        # --- question examples ---
        if "question_examples" in n:
            for q in n["question_examples"]:
                all_nodes.append({
                    "id": f"Question_{q['id']}",
                    "type": "QuestionExample",
                    "qtype_id": q.get("qtype_id", ""),
                    "question_text": q.get("question_text", ""),
                    "options": "; ".join(q.get("options", [])),
                    "answer": q.get("answer", ""),
                    "rationale": q.get("rationale", "")
                })

        # --- category children ---
        for cat, label in categories.items():
            if cat in n:
                for child in n[cat]:
                    child_node = {
                        "id": child["id"],
                        "type": label,
                        "description": child.get("description", "")
                    }
                    if "alias" in child:
                        child_node["alias"] = child["alias"]
                    if "examples" in child:
                        child_node["examples"] = "; ".join(child["examples"])
                    all_nodes.append(child_node)

                    if "subrules" in child:
                        for sub in child["subrules"]:
                            sub_node = {
                                "id": sub["id"],
                                "type": label + "Subrule",
                                "description": sub.get("description", "")
                            }
                            all_nodes.append(sub_node)

    return all_nodes


# ---- 6. Prepare relationships ----
def collect_rels():
    rels = []

    for n in nodes:
        parent = n["id"]

        if "format_rules" in n:
            for f in n["format_rules"]:
                rels.append({"from": parent, "to": f["id"], "rel": "HAS_FORMAT_RULE"})

        if "writing_styles" in n:
            for w in n["writing_styles"]:
                rels.append({"from": parent, "to": w["id"], "rel": "HAS_WRITING_STYLE"})

        if "skills" in n:
            for s in n["skills"]:
                rels.append({"from": parent, "to": s["id"], "rel": "HAS_SKILL"})

        if "distractors" in n:
            for d in n["distractors"]:
                rels.append({"from": parent, "to": d["id"], "rel": "HAS_DISTRACTOR"})

        if "penmanship" in n:
            for p in n["penmanship"]:
                rels.append({"from": parent, "to": p["id"], "rel": "HAS_PENMANSHIP_RULE"})
                if "subrules" in p:
                    for sub in p["subrules"]:
                        rels.append({"from": p["id"], "to": sub["id"], "rel": "HAS_SUBRULE"})

        if "question_types" in n:
            for q in n["question_types"]:
                rels.append({"from": parent, "to": q["id"], "rel": "HAS_QUESTION_TYPE"})

                if "skills" in q:
                    for s in q["skills"]:
                        rels.append({"from": q["id"], "to": s, "rel": "HAS_SKILL"})

                if "question_type_rules" in q:
                    for r in q["question_type_rules"]:
                        rels.append({"from": q["id"], "to": r, "rel": "HAS_RULE"})
        if "passage_examples" in n:
            for p in n["passage_examples"]:
                rels.append({
                    "from": parent,
                    "to": f"Passage_{p['id']}",
                    "rel": "HAS_PASSAGE_EXAMPLE"
                })

        if "question_examples" in n:
            for q in n["question_examples"]:
                rels.append({
                    "from": parent,
                    "to": f"Question_{q['id']}",
                    "rel": "HAS_QUESTION_EXAMPLE"
                })

                # Also link question → qtype
                if "qtype_id" in q and q["qtype_id"]:
                    rels.append({
                        "from": f"Question_{q['id']}",
                        "to": q["qtype_id"],
                        "rel": "OF_TYPE"
                    })


    # Add mid-level functional relations
    for l in links:
        if "from" in l and "to" in l and "relation" in l:
            rels.append({"from": l["from"], "to": l["to"], "rel": l["relation"]})

    return rels

# ---- 7. Insert into Neo4j ----
with driver.session() as session:
    # Nodes
    all_nodes = collect_nodes()
    session.execute_write(create_nodes_batch, all_nodes)

    # Relationships (grouped by type for efficiency)
    all_rels = collect_rels()
    rel_types = {}
    for r in all_rels:
        rel_types.setdefault(r["rel"], []).append(r)

    for rel_type, rows in rel_types.items():
        session.execute_write(create_rels_batch, rows, rel_type)

driver.close()
