# feedback.py

def validate_passage(passage: str):
    issues = []
    word_count = len(passage.split())
    if word_count < 700:
        issues.append("Passage too short (<700 words).")
    if passage.count("\n") < 3:
        issues.append("Too few paragraphs (<4).")
    if "Summary:" not in passage:
        issues.append("Missing summary line.")
    return issues


def validate_questions(questions_json, passage: str | None = None):
    """
    Validate a list of question objects.
    Expected fields: id, question_text, options, answer, rationale, linked_skills
    """
    issues = []
    for q in questions_json:
        qid = q.get("id", "<unknown>")
        if "answer" not in q or not q["answer"]:
            issues.append(f"Question {qid} missing correct answer.")
        if len(q.get("options", [])) < 4:
            issues.append(f"Question {qid} has fewer than 4 options.")
        if "question_text" not in q or not q["question_text"].strip():
            issues.append(f"Question {qid} missing question text.")
        if "rationale" not in q or not q["rationale"].strip():
            issues.append(f"Question {qid} missing rationale.")
        if "linked_skills" not in q or not q["linked_skills"]:
            issues.append(f"Question {qid} missing linked skills.")
        if passage and q.get("question_text") and q["question_text"] not in passage:
            # Light heuristic: check overlap of keywords
            if len(set(q["question_text"].lower().split()) & set(passage.lower().split())) < 3:
                issues.append(f"Question {qid} seems weakly connected to passage.")
    return issues


def validate_distractors(distractors_json):
    """
    Validate distractor patterns: each distractor should have
    - type
    - description
    - example
    """
    issues = []
    for i, d in enumerate(distractors_json, 1):
        if not d.get("type"):
            issues.append(f"Distractor {i} missing type.")
        if not d.get("description"):
            issues.append(f"Distractor {i} missing description.")
        if not d.get("example"):
            issues.append(f"Distractor {i} missing example.")
    return issues


def validate_penmanship(result_json):
    """
    Validate the output of check_penmanship executor.
    Expected structure: {"valid": bool, "violations": [...], "feedback": "..."}
    """
    issues = []
    if "valid" not in result_json:
        issues.append("Missing 'valid' field in penmanship result.")
    if "violations" not in result_json:
        issues.append("Missing 'violations' field in penmanship result.")
    if "feedback" not in result_json:
        issues.append("Missing 'feedback' field in penmanship result.")
    return issues


def build_feedback_examples(topic, passage, issues):
    """
    Build feedback examples for GEPA training.
    """
    return [{
        "input": topic,
        "output": passage[:200],
        "feedback": "; ".join(issues) if issues else "Looks good."
    }]
