# prompts.py - templates use placeholders {TOPIC}, {SOURCES}, {PASSAGE}

def format_sources(sources):
    """
    Convert sources into a clean, LLM-friendly format.
    Each source may have 'abstract' and 'facts'.
    Example:
    [S1] Abstract: ...
    [S1] Facts:
    - Fact 1
    - Fact 2
    """
    lines = []
    for s in sources:
        if s.get("abstract"):
            lines.append(f"[{s['id']}] Abstract: {s['abstract']}")
        if s.get("facts"):
            lines.append(f"[{s['id']}] Facts:")
            for f in s["facts"]:
                lines.append(f"- {f}")
    return "\n".join(lines)


# ---------- Passage Prompt Templates ----------

def build_passage_prompt_template():
    return """SYSTEM: You are an IELTS passage generator.

TOPIC: {TOPIC}

CONTEXT (external references):
{SOURCES}

TASK:
1) Use the KG (call the query_kg tool) to fetch constraints for node "IELTS_Academic_Reading" 
   (format rules, writing style, required skills).
2) Treat KG constraints as ground truth and follow them exactly.
3) Produce ONE IELTS-style Academic Reading passage (B2–C1), 4–6 paragraphs, ~900 words.
4) Use inline citations like [S1], [S2] whenever information comes from references.
5) End the passage with a line: Summary: <one-line summary>

OUTPUT: only the passage text (and Summary line).
"""


def build_passage_prompt(topic, sources, extra_context=None):
    """
    Returns a passage prompt string with {TOPIC} and {SOURCES} replaced.
    - sources: list of dicts {"id":..., "abstract":..., "facts":[...]}
    - extra_context: optional KG/system context
    """
    template = build_passage_prompt_template()
    sources_text = format_sources(sources)
    if extra_context:
        sources_text += "\n" + extra_context
    prompt = template.replace("{TOPIC}", topic)
    prompt = prompt.replace("{SOURCES}", sources_text)
    return prompt


# ---------- Question Prompt Templates ----------

def build_question_prompt_template(question_type="Multiple_Choice"):
    return ("""SYSTEM: You are an IELTS question generator.

PASSAGE:
{PASSAGE}

EXTERNAL CONTEXT (references + facts):
{SOURCES}

TASK:
1) Use the KG (call the query_kg tool) to fetch constraints for node related to question type '{qtype}'.
2) Produce 3 questions of type {qtype} that follow IELTS conventions:
   - Order follows passage order where applicable.
   - Each question: id, question_text, options (A/B/C/D), answer (single), rationale, linked_skills.
   - Distractors must follow KG distractor patterns.
OUTPUT: Return a JSON array of question objects.
""".replace("{qtype}", question_type))


def build_question_prompt(passage_text, question_type, sources=None, rules=None, skills=None, distractors=None):
    """
    Returns a question-generation prompt with placeholders filled.
    - passage_text: passage produced earlier
    - sources: list of dicts {"id":..., "abstract":..., "facts":[...]}
    - rules/skills/distractors: optional KG info (can be injected at the end)
    """
    template = build_question_prompt_template(question_type)

    # Format sources into readable text
    sources_text = format_sources(sources) if sources else "No external references provided."

    prompt = template.replace("{PASSAGE}", passage_text)
    prompt = prompt.replace("{SOURCES}", sources_text)

    # Add optional KG info (still separate from retriever context)
    context_parts = []
    if rules:
        context_parts.append("Rules:\n" + rules)
    if skills:
        context_parts.append("Skills:\n" + skills)
    if distractors:
        context_parts.append("Distractors:\n" + distractors)
    if context_parts:
        prompt += "\n\n" + "\n".join(context_parts)

    return prompt
