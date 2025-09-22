# validators.py
"""
Validators for GEPA (µ_f).
Extended with:
 - Penmanship scoring (from KG rules)
 - Writing style & cohesion checks
 - Distractor quality validation
 - Weighted composite score → IELTS band
 - Feedback example builder (from old feedback.py)
"""

import re
import json
from typing import Tuple, Dict, Any, List
from config import LLM_API_KEY, OPENAI_BASE_URL
from data_utils.json_strict import safe_json_loads
# ---------- Utilities ----------
def clean_passage_body(passage_text: str) -> str:
    """
    Remove metadata lines like 'Quiz title:', 'Quiz description:', etc.
    Keep only the labeled paragraphs and summary.
    """
    lines = passage_text.strip().splitlines()
    body_lines = [ln for ln in lines if ln.strip().startswith("Text:") or ln.strip().startswith("Summary:")]
    return "\n".join(body_lines)


def word_count(text: str) -> int:
    # Count words in cleaned passage body only
    body = clean_passage_body(text)
    return len(re.findall(r"\b\w+\b", body))


def paragraph_count(text: str) -> int:
    """
    Count how many paragraphs exist based on 'Text: [A-Z].' markers.
    Example: 'Text: A.' → counts as one paragraph.
    """
    body = clean_passage_body(text)
    matches = re.findall(r"Text:\s*[A-Z]\.", body)
    return len(matches)


def validate_passage_text(passage_text: str) -> Tuple[float, List[str], List[str]]:
    raw_traces = []
    fb_traces = []

    wc = word_count(passage_text)
    pc = paragraph_count(passage_text)

    # --- Word count scoring ---
    ideal = 900
    width = 900  # wider tolerance
    wc_score = max(0.0, 1 - abs(wc - ideal) / width)
    raw_traces.append(f"word_count={wc}")

    if wc < 600:
        fb_traces.append(f"Passage too short ({wc} words). Aim for ~900 words.")
    elif wc > 1200:
        fb_traces.append(f"Passage too long ({wc} words). Aim for ~900 words.")
    else:
        fb_traces.append(f"Passage length acceptable ({wc} words).")

    # --- Paragraph count ---
    if 5 <= pc <= 9:
        pc_score = 1.0
    else:
        pc_score = max(0.0, 1 - abs(pc - 7) / 7)
    raw_traces.append(f"paragraph_count={pc}")

    if pc < 5:
        fb_traces.append(f"Too few paragraphs ({pc}). Target 5–9.")
    elif pc > 9:
        fb_traces.append(f"Too many paragraphs ({pc}). Target 5–9.")
    else:
        fb_traces.append(f"Paragraph count within range ({pc}).")

    # --- Summary line ---
    if "Summary:" in passage_text:
        sum_score = 1.0
        raw_traces.append("summary=present")
        fb_traces.append("Summary line present at end.")
    else:
        sum_score = 0.5   # softer penalty
        raw_traces.append("summary=missing")
        fb_traces.append("Missing required summary line at end.")

    score = 0.5 * wc_score + 0.3 * pc_score + 0.2 * sum_score
    return score, raw_traces, fb_traces

# ---------- Question validator ----------
def validate_questions_structure(questions_list) -> Tuple[float, List[str], List[str]]:
    raw_traces = []
    fb_traces = []

    if not isinstance(questions_list, list) or not questions_list:
        return 0.3, ["questions=missing_or_not_list"], [
            "Questions missing or invalid JSON. Require a valid JSON array of questions."
        ]

    total_q = len(questions_list)
    ok_count = 0
    for q in questions_list:
        if not q.get("id") or not q.get("question_text"):
            raw_traces.append(f"question_missing_fields:{q.get('id','?')}")
            fb_traces.append("Some questions missing ID or text → ensure each has 'id' and 'question_text'.")
            continue
        if "answer" not in q or q["answer"] is None:
            raw_traces.append(f"question_{q.get('id')} missing_answer")
            fb_traces.append(f"Question {q.get('id','?')} missing answer → always include 'answer'.")
            continue
        ok_count += 1

    score = ok_count / total_q if total_q else 0.3
    if score < 1.0:
        fb_traces.append(f"Only {ok_count}/{total_q} questions valid. Ensure all have complete fields.")
    else:
        fb_traces.append("All questions valid and well-structured.")

    return score, raw_traces, fb_traces


# ---------- Extractive check ----------
def extractive_answer_check(passage_text: str, question) -> Tuple[float, str]:
    ans = question.get("answer", "")
    if not ans:
        return 0.0, "answer_empty"
    ans_lower = ans.lower()
    if ans_lower in passage_text.lower():
        return 1.0, "answer_span_found"
    words = [w for w in re.findall(r"\w+", ans_lower) if len(w) > 3]
    if words and all(w in passage_text.lower() for w in words):
        return 0.75, "answer_words_all_present"
    return 0.0, "answer_missing_or_paraphrased"


# ---------- Penmanship validator ----------
def validate_penmanship(text: str, rules: List[Dict] | None = None) -> Tuple[float, List[str], List[str]]:
    raw_traces = []
    fb_traces = []

    if not rules:
        return 1.0, ["penmanship=skipped(no_rules)"], ["No penmanship rules provided."]

    violations = []
    for rule in rules:
        desc = rule.get("description", "")
        patterns = rule.get("banned_patterns", [])
        for pat in patterns:
            if re.search(pat, text):
                violations.append(desc)
                raw_traces.append(f"penmanship_violation:{desc}")
                fb_traces.append(f"Penmanship violation: {desc}")

    score = 1.0 if not violations else max(0.0, 1 - len(violations) / len(rules))
    if not violations:
        fb_traces.append("No penmanship violations detected.")
    return score, raw_traces, fb_traces


# ---------- Distractor quality validator ----------
def validate_distractors(questions: list) -> Tuple[float, List[str], List[str]]:
    raw_traces = []
    fb_traces = []

    if not questions:
        return 0.0, ["distractors=missing"], ["No distractors found. Require at least 2–3 per question."]

    valid = 0
    total = 0
    for q in questions:
        opts = q.get("options", [])
        ans = q.get("answer", "")
        for opt in opts:
            if opt == ans:
                continue
            total += 1
            if abs(len(opt) - len(ans)) < 10:  # rough similarity
                valid += 1
            else:
                raw_traces.append(f"distractor_bad_length:{opt}")
                fb_traces.append(f"One distractor too different in length: '{opt}'")

    score = valid / total if total else 0.0
    if score >= 0.7:
        fb_traces.append(f"Distractor quality good ({valid}/{total} acceptable).")
    else:
        fb_traces.append(f"Distractor quality weak ({valid}/{total} acceptable). Balance lengths with correct answer.")

    return score, raw_traces, fb_traces



# ---------- Band mapping ----------
def to_band(score_01: float) -> float:
    band = score_01 * 9.0
    return round(band * 2) / 2

def validate_by_llm(passage: str) -> Dict[str, Any]:
    """
    Call LLM to evaluate passage similarity with IELTS style.
    Returns dict with category scores + overall + feedback.
    """
    from openai import OpenAI
    import json

    client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE_URL)

    IELTS_EVAL_PROMPT = """IELTS Reading Passage Evaluation Prompt (Single Passage)

    System Role:
    You are an IELTS Reading Passage Validator. Your task is to evaluate how closely a single passage resembles an authentic IELTS Reading passage, based on specific categories used in IELTS design.

    🔎 Categories to Evaluate
    1. Vocabulary Level (0–10)

    Uses Academic Word List (AWL) and common academic vocabulary.

    Rare/technical words ≤ 5%, and jargon is defined if used.

    Avoids overly literary or archaic terms.

    Target level: CEFR B2–C1.

    2. Sentence Length & Grammar Complexity (0–10)

    Average sentence length: 15–25 words.

    Maximum sentence length: ≤ 35 words.

    Mix of simple, compound, and complex sentences (not all long and dense).

    Make sure complex sentences only accounts for 55-60% the passage.

    Limited subordinate clauses (≤2 per sentence).

    3. Readability (0–10)

    Flesch Reading Ease (FRE): Ideal: 40–60.

    Flesch–Kincaid Grade Level (FKGL): Ideal: 9–11.

    Smooth flow without excessive nominalisation or passive voice.

    4. Content Balance (0–10)

    Mix of facts, explanations, and some discussion.

    Provides examples, statistics, or references (e.g., organizations, studies).

    Neutral, informative tone (not persuasive or emotional).

    5. Authenticity of Style (0–10)

    Formal, academic but accessible.

    Resembles Cambridge IELTS passage style.

    Avoids journalistic flair, literary metaphors, or conversational tone.
    FOLLOW THIS Output Format: (JSON)
{{
  "Vocabulary_Level": <score>,
  "Sentence_Length_&_Grammar_Complexity": <score>,
  "Readability": <score>,
  "Content_Balance": <score>,
  "Authenticity_of_Style": <score>,
  "Feedbacks": {{
    "Vocabulary_Level": "...",
    "Sentence_Length_&_Grammar_Complexity": "...",
    "Readability": "...",
    "Content_Balance": "...",
    "Authenticity_of_Style": "..."
  }}
}}
    Passage:
    \"\"\"{passage}\"\"\"
    """.format(passage=passage)

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "developer", "content": "You are an IELTS Reading examiner assistant."},
                  {"role": "user", "content": IELTS_EVAL_PROMPT}],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict) and parsed.get("__parse_error"):
        # keep old fallback numbers (existing behavior)
        result = {
            "Vocabulary_Level": 0.0,
            "Sentence_Length_&_Grammar_Complexity": 0.0,
            "Readability": 0.0,
            "Content_Balance": 0.0,
            "Authenticity_of_Style": 0.0,
            "Feedbacks": {
                "Vocabulary_Level": "...",
                "Sentence_Length_&_Grammar_Complexity": "...",
                "Readability": "...",
                "Content_Balance": "...",
                "Authenticity_of_Style": "..."
            }
        }
    else:
        result = parsed
    return result


# ---------- Composer ----------
def score_passage_and_questions(outputs: Dict[str, Any], topic: str,
                                penmanship_rules: List[Dict] | None = None) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    raw_traces = []
    fb_traces = []
    passage = outputs.get("passage", "")
    questions_raw = outputs.get("questions", "")

    # Parse questions if JSON string
    questions = questions_raw
    if isinstance(questions_raw, str):
        try:
            m = re.search(r'(\{.*\}|\[.*\])', questions_raw, flags=re.S)
            if m:
                parsed = safe_json_loads(m.group(1))
                if isinstance(parsed, dict) and parsed.get("__parse_error"):
                    # handle as failure (previous logic might have continued)
                    questions = None  # or fallback behavior as your code expects
                else:
                    questions = parsed
            else:
                questions = None

        except Exception:
            questions = []

    # --- Sub-scores with extended validators ---
    p_score, p_raw, p_fb = validate_passage_text(passage)
    q_score, q_raw, q_fb = validate_questions_structure(questions)
    # --- New LLM validator ---
    llm_scores = validate_by_llm(passage)

    # --- Collect raw traces (debug) ---
    raw_traces += [f"P:{t}" for t in p_raw]
    raw_traces += [f"Q:{t}" for t in q_raw]
    # raw_traces += [f"PN:{t}" for t in pn_raw]
    # raw_traces += [f"D:{t}" for t in d_raw]
    for k, v in llm_scores['Feedbacks'].items():
        raw_traces.append(f"LLM:{k}={v}")



    # --- Collect feedback traces (for GEPA mutation) ---
    fb_traces += [f"P:{t}" for t in p_fb]
    fb_traces += [f"Q:{t}" for t in q_fb]
    # fb_traces += [f"PN:{t}" for t in pn_fb]
    # fb_traces += [f"D:{t}" for t in d_fb]
    for k, v in llm_scores['Feedbacks'].items():
        raw_traces.append(f"LLM:{k}={v}")
    # --- Extractive check ---
    extract_scores = []
    for q in questions:
        s, trace = extractive_answer_check(passage, q)
        extract_scores.append(s)
        raw_traces.append(f"EX:{q.get('id','?')}:{trace}")
        fb_traces.append(f"Answer validation for {q.get('id','?')}: {trace}")
    extract_avg = sum(extract_scores) / len(extract_scores) if extract_scores else 0.0

    # --- Scores dict ---
    scores = {
        "passage": p_score,
        "questions": q_score,
        # "distractors": distractor_score,
        "extractive": extract_avg,
        "Vocabulary_Level": llm_scores["Vocabulary_Level"] / 10.0,
        "Sentence_Length_&_Grammar_Complexity": llm_scores["Sentence_Length_&_Grammar_Complexity"] / 10.0,
        "Authenticity_of_Style": llm_scores["Authenticity_of_Style"] / 10.0,
        "Content_Balance": llm_scores["Content_Balance"] / 10.0,
        "Readability": llm_scores["Readability"] / 10.0
    }

    # --- Final weighted score ---
    final_score = (
        0.20 * p_score +
        0.15 * q_score +
        0.05 * extract_avg +
        0.15 * llm_scores["Vocabulary_Level"] / 10.0 +
        0.10 * llm_scores["Sentence_Length_&_Grammar_Complexity"] / 10.0 + 
        0.10 * llm_scores["Readability"] / 10.0 + 
        0.05 * llm_scores["Content_Balance"] / 10.0 + 
        0.20 * llm_scores["Authenticity_of_Style"] / 10.0
        # + 0.10 * distractor_score  (enable later if distractors are required)
    )
    band = to_band(final_score)

    raw_traces.append(f"SCORE_BAND={band}")
    fb_traces.append(f"Overall estimated IELTS band: {band} (0–9 scale).")
    scores = {k: float(v) for k, v in scores.items()}
    return scores, {"raw": raw_traces, "feedback": fb_traces}


# ---------- Composer ----------
def score_passages_only(outputs: Dict[str, Any], topic: str,
                                penmanship_rules: List[Dict] | None = None) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    raw_traces = []
    fb_traces = []
    passage = outputs.get("passage", "")


    # --- Sub-scores with extended validators ---
    p_score, p_raw, p_fb = validate_passage_text(passage)
    # --- New LLM validator ---
    llm_scores = validate_by_llm(passage)

    # --- Collect raw traces (debug) ---
    raw_traces += [f"P:{t}" for t in p_raw]
    # raw_traces += [f"PN:{t}" for t in pn_raw]
    # raw_traces += [f"D:{t}" for t in d_raw]
    for k, v in llm_scores['Feedbacks'].items():
        raw_traces.append(f"LLM:{k}={v}")



    # --- Collect feedback traces (for GEPA mutation) ---
    fb_traces += [f"P:{t}" for t in p_fb]
    # fb_traces += [f"PN:{t}" for t in pn_fb]
    # fb_traces += [f"D:{t}" for t in d_fb]
    for k, v in llm_scores['Feedbacks'].items():
        raw_traces.append(f"LLM:{k}={v}")

    # --- Scores dict ---
    scores = {
        "passage": p_score,
        # "distractors": distractor_score,
        "Vocabulary_Level": llm_scores["Vocabulary_Level"] / 10.0,
        "Sentence_Length_&_Grammar_Complexity": llm_scores["Sentence_Length_&_Grammar_Complexity"] / 10.0,
        "Authenticity_of_Style": llm_scores["Authenticity_of_Style"] / 10.0,
        "Content_Balance": llm_scores["Content_Balance"] / 10.0,
        "Readability": llm_scores["Readability"] / 10.0
    }

    # --- Final weighted score ---
    final_score = (
        0.20 * p_score +
        0.15 * llm_scores["Vocabulary_Level"] / 10.0 +
        0.15 * llm_scores["Sentence_Length_&_Grammar_Complexity"] / 10.0 + 
        0.10 * llm_scores["Readability"] / 10.0 + 
        0.10 * llm_scores["Content_Balance"] / 10.0 + 
        0.20 * llm_scores["Authenticity_of_Style"] / 10.0
        # + 0.10 * distractor_score  (enable later if distractors are required)
    )
    band = to_band(final_score)

    raw_traces.append(f"SCORE_BAND={band}")
    fb_traces.append(f"Overall estimated IELTS band: {band} (0–9 scale).")
    scores = {k: float(v) for k, v in scores.items()}
    return scores, {"raw": raw_traces, "feedback": fb_traces}


# ---------- Feedback Examples ----------
def build_feedback_examples(topic: str, passage: str, issues: List[str]) -> List[Dict[str, str]]:
    return [{
        "input": topic,
        "output": passage[:200],
        "feedback": "; ".join(issues) if issues else "Looks good."
    }]
