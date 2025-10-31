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
import time
import re
import json
from typing import Tuple, Dict, Any, List
from config import LLM_API_KEY, OPENAI_BASE_URL, PART_USED
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

    ideal_level = 900 if PART_USED < 4 else 300
    # --- Word count scoring ---
    ideal = ideal_level
    width = ideal_level  # wider tolerance
    wc_score = max(0.0, 1 - abs(wc - ideal) / (width * 2.5))
    raw_traces.append(f"word_count={wc}")

    if wc < ideal_level * 2 / 3:
        fb_traces.append(f"Passage too short ({wc} words). Aim for ~{ideal_level} words.")
    elif wc > ideal_level * 4 / 3:
        fb_traces.append(f"Passage too long ({wc} words). Aim for ~{ideal_level} words.")
    else:
        fb_traces.append(f"Passage length acceptable ({wc} words).")

    # --- Paragraph count ---
    ideal_paragraphs = 7 if PART_USED < 4 else 5
    if ideal_paragraphs - ideal_paragraphs // 3 <= pc <= ideal_paragraphs + ideal_paragraphs // 3:
        pc_score = 1.0
    else:
        pc_score = max(0.0, 1 - abs(pc - ideal_paragraphs) / ideal_paragraphs)
    raw_traces.append(f"paragraph_count={pc}")

    if pc < ideal_paragraphs - ideal_paragraphs // 3:
        fb_traces.append(f"Too few paragraphs ({pc}). Target {ideal_paragraphs - ideal_paragraphs // 3} to {ideal_paragraphs + ideal_paragraphs // 3}.")
    elif pc > ideal_paragraphs + ideal_paragraphs // 3:
        fb_traces.append(f"Too many paragraphs ({pc}). Target {ideal_paragraphs - ideal_paragraphs // 3} to {ideal_paragraphs + ideal_paragraphs // 3}.")
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




# ---------- Band mapping ----------
def to_band(score_01: float) -> float:
    band = score_01 * 9.0
    return round(band * 2) / 2


IELTS_EVAL_PROMPTS = {
    1: """IELTS Reading Passage Evaluation Prompt (Single Passage — Part 1)

System Role:
You are an IELTS Reading Passage Validator. Your task is to evaluate how closely a passage resembles an authentic IELTS Reading Part 1 passage, based on the categories below.

🔎 Categories to Evaluate

1. Vocabulary Level (0–100)
- ~85–90% GSL (basic words): clear, everyday vocabulary.
- ~8–10% AWL (academic or semi-formal words): mild sophistication.
- ≤2% technical or field-specific terms, only if clearly explained in context.
- Avoids idioms, figurative language, or uncommon expressions.
- Target level: CEFR B1 (borderline B2) — simple but precise.

2. Sentence Length & Grammar Complexity (0–100)
- Average sentence length: 10–18 words.
- Maximum sentence length: ≤28 words.
- Predominantly simple and compound sentences.
- Complex sentences ≤35–40% of text.
- Minimal use of subordination and nominalisation.
- Grammar and punctuation are clear and functional — not dense or abstract.

3. Readability (0–100)
- FRE: 55–70 (fairly easy to read).
- FKGL: 7–8.
- Direct, instructional, or descriptive tone — no academic abstraction.
- Uses active voice and clear connectors (“first,” “next,” “for example”).
- Paragraphs flow logically and are easy to follow without rereading.

4. Content Balance (0–100)
- Everyday, workplace, or public information context (e.g., notices, instructions, guides, brochures).
- Focus on facts, procedures, or practical advice.
- Limited inference required — answers often found directly in text.
- Neutral, helpful, and informative, not argumentative or opinion-based.
- Includes enough detail for comprehension but avoids data-heavy analysis.

5. Authenticity of Style (0–100)
- Style resembles Cambridge IELTS Part 1 passages: clear, factual, and purpose-driven.
- Tone is neutral, informative, and slightly formal, but not academic.
- Avoids storytelling, persuasion, or expressive style.
- Feels like a real-world information text (e.g., workplace memo, museum leaflet, company policy excerpt).

FOLLOW THIS Output Format (JSON):

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
\"\"\"{passage}\"\"\"""",

    2: """IELTS Reading Passage Evaluation Prompt (Single Passage — Part 2)

System Role:
You are an IELTS Reading Passage Validator. Your task is to evaluate how closely a Part 2 passage resembles an authentic IELTS Reading Part 2 passage, based on the categories below.

🔎 Categories to Evaluate

1. Vocabulary Level (0–100)
~70–75% GSL (basic words): ensures accessibility.
~20–25% AWL (academic vocabulary): adds professional/academic tone.
~5% technical/workplace terms: linked to training, management, or procedures.
Avoids rare/literary terms.
Target level: CEFR upper B1–B2 (borderline C1 in places).

2. Sentence Length & Grammar Complexity (0–100)
Average sentence length: 14–22 words.
Maximum sentence length: ≤32 words.
Mix of simple, compound, and complex sentences.
Complex sentences ~50–55% of text.
Subordinate clauses: mostly ≤2 per sentence.
Style slightly denser than Part 1, but less abstract than Part 3.

3. Readability (0–100)
FRE: 45–60 (harder than Part 1, but not as dense as Part 3).
FKGL: 9–10.
Some nominalisation and passive voice allowed if natural in workplace/academic context.
Flow is structured, concise, but not conversational.

4. Content Balance (0–100)
Workplace or training context with semi-academic style.
Mix of policy/rules, explanations, and implications.
May cite organisations, processes, or short case examples.
Informative, neutral, factual — avoids persuasion or narrative.
Enough detail to require close reading (not skim-level like Part 1).

5. Authenticity of Style (0–100)
Clear, professional, and semi-academic.
Resembles Cambridge IELTS Part 2 passages (e.g., workplace manuals, HR policies, training documents, short reports).
Avoids journalistic flair, metaphors, or casual tone.
Concise, objective, slightly formal.

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
\"\"\"{passage}\"\"\"""",

    3: """IELTS Reading Passage Evaluation Prompt (Single Passage — Part 3)

System Role:
You are an IELTS Reading Passage Validator. Your task is to evaluate how closely a passage resembles an authentic IELTS Reading Part 3 passage, based on the categories below.

🔎 Categories to Evaluate

1. Vocabulary Level (0–100)
- ~60–65% GSL (general words).
- ~25–30% AWL (academic vocabulary).
- ~5–10% technical or field-specific terms (in context of psychology, sociology, science, or education).
- Avoids obscure or literary words.
- Target level: CEFR B2–C1.

2. Sentence Length & Grammar Complexity (0–100)
- Average sentence length: 18–26 words.
- Maximum sentence length: ≤35 words.
- High ratio of complex sentences (~60–65%).
- More frequent subordination and nominalisation, but remains clear.
- Logical connectors (“however”, “in contrast”, “therefore”) used precisely.
- Style dense, academic, and argumentative.

3. Readability (0–100)
- FRE: 35–50.
- FKGL: 11–12.
- Frequent use of abstract nouns, passives, and cohesive devices.
- Demands inference and analysis.
- Flow is academic, precise, and structured.

4. Content Balance (0–100)
- Academic or semi-academic text (psychology, history, science, or social studies).
- Focus on explanation, comparison, or argumentation.
- Includes evidence, studies, or researcher names.
- Requires interpretation or critical understanding.

5. Authenticity of Style (0–100)
- Closely resembles Cambridge IELTS Part 3 style: formal, academic, and analytic.
- No narrative, advertising, or instructional tone.
- Balanced and impersonal, presenting multiple viewpoints.

FOLLOW THIS Output Format (JSON):

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
\"\"\"{passage}\"\"\"""",

    4: """Secondary Reading Passage Evaluation Prompt (Single Passage)

System Role:
You are a Secondary Reading Passage Validator. Your task is to evaluate how closely a passage resembles an appropriate academic-style reading text for secondary students (easier than IELTS Part 1), based on the categories below.

🔎 Categories to Evaluate

1. Vocabulary Level (0–100)
- 80–85% GSL (basic, high-frequency words).
- 10–15% AWL (academic vocabulary for enrichment).
- ≤5% technical terms, and these must be explained in the text.
- Rare / low-frequency words (<5000 frequency rank) must be <3% of total text.
- Avoids literary, idiomatic, or overly abstract words.
- Target level: CEFR mid B1.

2. Sentence Length & Grammar Complexity (0–100)
- Average sentence length: 12–18 words.
- Maximum sentence length: ≤26 words.
- Mostly simple and compound sentences; some complex (~30–35%).
- Subordinate clauses: rarely more than 1–2 per sentence.
- Style accessible, but slightly denser than everyday reading.

3. Readability (0–100)
- FRE: 55–70 (clearer than IELTS Part 1).
- FKGL: 7–8.
- Mix of straightforward narration + semi-academic phrasing.
- Limited nominalisation and passive voice, used only where natural.

4. Content Balance (0–100)
- Topics: cultural, historical, educational, or social (e.g., traditions, food, inventions, discoveries).
- Explanatory and informative, with examples or case studies.
- Avoids abstract theory, heavy statistics, or workplace reports.
- Neutral, factual, and balanced.

5. Authenticity of Style (0–100)
- Resembles simplified academic articles or school-level non-fiction.
- Objective, neutral, formal but readable.
- No journalistic flair, no storytelling fiction, no persuasive tone.
- Suitable for training secondary students in academic reading.

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
\"\"\"{passage}\"\"\""""
}

def validate_by_llm(passage: str) -> Dict[str, Any]:
    """
    Call LLM to evaluate passage similarity with IELTS style.
    Returns dict with category scores + overall + feedback.
    """
    from openai import OpenAI
    import json

    client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE_URL)
    IELTS_EVAL_PROMPT  = IELTS_EVAL_PROMPTS[PART_USED].format(passage=passage)


    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "developer", "content": "You are an IELTS Reading examiner assistant."},
                  {"role": "user", "content": IELTS_EVAL_PROMPT}] if PART_USED < 4 
                  else[{"role": "developer", "content": "You are a Simplified IELTS Reading Passage examiner assistant (Secoondary Student level Passage)."},
                       {"role": "user", "content": IELTS_EVAL_PROMPT}]
                  ,

        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()
    time.sleep(2)  # avoid rate limits
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
        fb_traces.append(f"LLM:{k}={v}")
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
        "Vocabulary_Level": llm_scores["Vocabulary_Level"] / 100.0,
        "Sentence_Length_&_Grammar_Complexity": llm_scores["Sentence_Length_&_Grammar_Complexity"] / 100.0,
        "Authenticity_of_Style": llm_scores["Authenticity_of_Style"] / 100.0,
        "Content_Balance": llm_scores["Content_Balance"] / 100.0,
        "Readability": llm_scores["Readability"] / 100.0
    }

    # --- Final weighted score ---
    final_score = (
        0.20 * p_score +
        0.15 * q_score +
        0.05 * extract_avg +
        0.15 * llm_scores["Vocabulary_Level"] / 100.0 +
        0.10 * llm_scores["Sentence_Length_&_Grammar_Complexity"] / 100.0 + 
        0.10 * llm_scores["Readability"] / 100.0 + 
        0.05 * llm_scores["Content_Balance"] / 100.0 + 
        0.20 * llm_scores["Authenticity_of_Style"] / 100.0
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
        fb_traces.append(f"LLM:{k}={v}")

    # --- Scores dict ---
    scores = {
        "passage": p_score,
        # "distractors": distractor_score,
        "Vocabulary_Level": llm_scores["Vocabulary_Level"] / 100.0,
        "Sentence_Length_&_Grammar_Complexity": llm_scores["Sentence_Length_&_Grammar_Complexity"] / 100.0,
        "Authenticity_of_Style": llm_scores["Authenticity_of_Style"] / 100.0,
        "Content_Balance": llm_scores["Content_Balance"] / 100.0,
        "Readability": llm_scores["Readability"] / 100.0
    }

    # --- Final weighted score ---
    final_score = (
        0.20 * p_score +
        0.15 * llm_scores["Vocabulary_Level"] / 100.0 +
        0.15 * llm_scores["Sentence_Length_&_Grammar_Complexity"] / 100.0 + 
        0.10 * llm_scores["Readability"] / 100.0 + 
        0.10 * llm_scores["Content_Balance"] / 100.0 + 
        0.20 * llm_scores["Authenticity_of_Style"] / 100.0
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
