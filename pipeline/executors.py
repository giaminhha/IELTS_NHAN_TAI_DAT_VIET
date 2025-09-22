# executors.py
"""
Upgraded executors:
 - Uses MCPClient to fetch KG constraints (cached MCP endpoints).
 - Strict prompt templates with constraints.
 - Validate + retry loop using validators.py.
 - Backwards-compatible executor signatures used by gepa_driver / main.py:
     executor(prompt_template, topic, outputs_so_far) -> returns str or structured object
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from data_utils.json_strict import safe_json_loads
from mcp_integration.mcp_client import MCPClient
from pipeline.llm import _extract_json_from_text
# Try to import your llm helper (project-specific). If not present, we'll raise helpful errors.
try:
    from pipeline.llm import call_llm, DEBUG_STUB  # call_llm(prompt, expect_json=False, max_tokens=...)
except Exception:
    # Minimal fallback placeholder to avoid crashes during static analysis.
    DEBUG_STUB = True
    def call_llm(prompt: str, expect_json: bool = False, max_tokens: int = 10000):
        raise RuntimeError("llm.call_llm not available. Please provide llm.call_llm in your project.")

# Import validators provided in your repo
try:
    from pipeline.validators import validate_passage_text, validate_questions_structure
except Exception:
    # If validators.py is missing, create a minimal fallback so errors are clear.
    def validate_passage_text(txt):
        return 0.0, ["validators.validate_passage_text unavailable"]
    def validate_questions_structure(qs):
        return 0.0, ["validators.validate_questions_structure unavailable"]

# configure logger
logger = logging.getLogger("executors")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

# Instantiate MCP client (adjust URL if your MCP runs elsewhere)
MCP = MCPClient("http://localhost:8000")

# -----------------------
EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_passage_examples",
            "description": "Retrieve example IELTS reading passages from the KG",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": "The ID of the test node, e.g., 'IELTS_Academic_Reading'"
                    }
                },
                "required": ["test_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_question_examples",
            "description": "Retrieve example IELTS reading questions from the KG",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": "The ID of the test node, e.g., 'IELTS_Academic_Reading'"
                    },
                    "qtype_id": {
                        "type": "string",
                        "description": "Optional. The ID of the question type to filter by, e.g., 'Multiple_Choice'"
                    }
                },
                "required": ["test_id"]
            }
        }
    }
]


PASSAGE_TOOLS = [EXAMPLE_TOOLS[0]]    # passage examples
QUESTION_TOOLS = [EXAMPLE_TOOLS[1]]   # question examples


def maybe_call_examples(model_out: str, tool_group: str, qtype_id: Optional[str] = None):
    """
    If the model requests example tool, fetch from MCP and continue generation.
    Otherwise return the original output.
    """
    try:
        parsed = safe_json_loads(model_out)
        if isinstance(parsed, dict) and "tool" in parsed:
            if parsed["tool"] == "get_passage_examples":
                examples = MCP._get("/get_passage_examples")
                followup = f"Here are example passages:\n{json.dumps(examples, indent=2)}\n\nNow continue your task."
                return call_llm(followup, max_tokens=2500)
            elif parsed["tool"] == "get_question_examples":
                endpoint = f"/get_question_examples/{qtype_id}"
                examples = MCP._get(endpoint)
                followup = f"Here are example questions:\n{json.dumps(examples, indent=2)}\n\nNow continue your task."
                return call_llm(followup, max_tokens=2500)
    except Exception:
        return model_out
    return model_out
# -----------------------
# Helpers
# -----------------------
def _strip_code_fence(text: str) -> str:
    # remove triple-backtick fenced blocks if the model wrapped JSON in them
    if text.strip().startswith("```"):
        parts = text.strip().split("```", 2)
        # If there are 3 parts, the middle is the fenced content
        if len(parts) >= 3:
            return parts[1].strip()
    return text

def _parse_json_from_model(text: str) -> Any:
    """
    Try to robustly parse JSON returned by LLM.
    - strips code fences
    - tries json.loads
    - if fails, attempts to find first '{' or '[' and parse substring
    """
    txt = _strip_code_fence(text).strip()
    parsed = safe_json_loads(txt)
    if isinstance(parsed, dict) and parsed.get("__parse_error"):
        # attempt to find first JSON-like substring
        first = min(
            (txt.find("["), txt.find("{"))
            if "[" in txt or "{" in txt
            else (len(txt), len(txt))
        )
        if first != -1 and first < len(txt):
            sub = txt[first:]
            try:
                nw_parsed = safe_json_loads(sub)
                if isinstance(nw_parsed, dict) and nw_parsed.get("__parse_error"):
                    try:
                        return json.loads(sub)  # final direct attempt
                    except Exception as e:
                        logger.debug("JSON sub-parse failed: %s", e)
                        return {"raw": sub, "parse_error": nw_parsed["__parse_error"]}
                return nw_parsed
            except Exception as e:
                logger.debug("JSON salvage failed: %s", e)
                return {"raw": txt, "parse_error": parsed["__parse_error"]}
        else:
            return {"raw": txt, "parse_error": parsed["__parse_error"]}
    else:
        return parsed


def _short(x: str, n: int = 400) -> str:
    return x if len(x) <= n else x[:n] + " ..."

# -----------------------
# Prompt templates (strict)
# -----------------------
part_generated = 1
with open(f"C:/Users/Dell/Downloads/IELTS_NHAN_TAI_DAT_VIET/template/PASSAGE_TEMPLATE_PART_{part_generated}.txt", "r", encoding="utf-8") as f:
    _PASSAGE_TEMPLATE = f.read()


part_generated = 1
with open(f"C:/Users/Dell/Downloads/IELTS_NHAN_TAI_DAT_VIET/template/QUESTIONS_TEMPLATE_PART_{part_generated}.txt", "r", encoding="utf-8") as f:
    _QUESTION_TEMPLATE = f.read()


_DISTRACTOR_TEMPLATE = """SYSTEM: You are an IELTS distractor generator.

PASSAGE:
{passage}

KG DISTRACTOR PATTERNS:
{d_rules}

EXAMPLE SUMMARIES OF DISTRACTOR TYPES:
- Lexical Similarity: Wrong option looks similar in wording (synonyms, paraphrased terms).
- Plausible but Wrong Detail: Option mentions something present in passage but not correct for the question.
- Outside Knowledge Trap: Option is plausible using world knowledge but not supported in passage.
- Opposite/Contradiction: Option states the reverse of what passage says.
- Irrelevant but Related Topic: Option is thematically related but not directly answering.
- Overgeneralisation: Option uses extreme or absolute wording not supported by passage.

TASK:
1) For each question (or for the passage in general), produce a short list of distractors following KG patterns.
2) Output a JSON array of objects: 
   {{ "for_question_id": "...", 
   "distractors": [ {{ "text": "...", "pattern": "..." }}, ... ] }}

"""


# -----------------------
# Executors (public)
# -----------------------
def passage_executor(prompt_template: str, topic: str, outputs_so_far: Dict[str, Any]) -> str:
    """
    Signature kept for compatibility with GEPA orchestration:
      (prompt_template, topic, outputs_so_far)
    Behavior: fetch sources (from outputs_so_far['sources'] if present, else call retriever inside),
              fetch KG passage rules from MCP, generate and validate the passage (with retries).
    Returns: passage text (string).
    """
    # sources: prefer those passed in from main/orchestrator
    sources = outputs_so_far.get("sources")
    if not sources:
        # lazy import retriever here so executors can be imported standalone
        try:
            from pipeline.retriever import retrieve_sources
            sources = retrieve_sources(topic)
        except Exception:
            sources = []
            logger.warning("retriever.retrieve_sources unavailable; proceeding without sources.")

    # flatten sources for prompt
    sources_txt = "\n".join([f"[S{i+1}] {s.get('title','')}. {s.get('abstract', s.get('text',''))}" 
                              for i, s in enumerate(sources[:6])])

    kg_rules = MCP.get_passage_rules() + MCP.get_penmanship_rules()
    kg_rules_txt = json.dumps(kg_rules, ensure_ascii=False, indent=2)

    # attempts + validation loop
    max_attempts = 3
    threshold = 0.70  # passage validator score threshold
    last_traces = None

    for attempt in range(1, max_attempts + 1):
        logger.info("PASSAGE_GENERATE attempt %d/%d for topic=%s", attempt, max_attempts, topic)
        prompt = _PASSAGE_TEMPLATE.format(topic=topic, sources=sources_txt, kg_rules=kg_rules_txt)

        if DEBUG_STUB:
            # useful for local debug runs
            passage = f"DEBUG PASSAGE about {topic}\n\nSummary: stub."
        else:
            passage = call_llm(prompt, tools=PASSAGE_TOOLS, expect_json=False, max_tokens=10000)
            passage = maybe_call_examples(passage, "fetch_passage_examples")

        # validate
        score, raw_traces, fb_traces = validate_passage_text(passage)
        logger.info("Passage score=%.3f raw=%s feedback=%s", score, raw_traces, fb_traces)
        last_traces = raw_traces

        if score >= threshold:
            return passage

        fix_instructions = "Please regenerate the passage and fix: " + "; ".join(fb_traces)
        prompt += "\n\nFEEDBACK: " + fix_instructions


        # small wait/backoff to avoid rate-limit spikes
        time.sleep(5)

    # After attempts exhausted: return last passage but mark in logs
    logger.warning("PASSAGE generation failed to meet threshold after %d attempts. Returning last result.", max_attempts)
    return "Failure Passage"


def questions_executor(prompt_template: str, topic: str, outputs_so_far: Dict[str, Any], qtype_id: str = "Multiple_Choice", count: int = 3) -> List[Dict]:
    """
    Generate questions for a given passage present in outputs_so_far['passage'] (or raise).
    Returns: list of question dicts (if JSON parsing fails, returns an empty list or raw text).
    """
    passage = outputs_so_far.get("passage")
    if not passage:
        raise ValueError("questions_executor requires 'passage' in outputs_so_far")

    qtype_rules = MCP.get_distractor_patterns() + MCP.get_question_type_context(qtype_id) 
    qtype_rules_txt = json.dumps(qtype_rules, ensure_ascii=False, indent=2)

    max_attempts = 2
    threshold = 0.80  # question structure acceptance

    for attempt in range(1, max_attempts + 1):
        logger.info("QUESTIONS_GENERATE attempt %d/%d qtype=%s", attempt, max_attempts, qtype_id)
        prompt = _QUESTION_TEMPLATE.format(passage=passage, qtype_rules=qtype_rules_txt, qtype_id=qtype_id, count=count)

        if DEBUG_STUB:
            model_out = json.dumps([
                {"id":"Q1","question_text":"DEBUG Q?","options":["A","B","C","D"],"answer":"A","rationale":"stub","linked_skills":["Skimming"]}
            ])
        else:
            model_out = call_llm(prompt, tools=QUESTION_TOOLS, expect_json=False, max_tokens=10000)
            model_out = maybe_call_examples(model_out, "fetch_questions_examples", qtype_id=qtype_id)

        try:
            parsed = _parse_json_from_model(model_out)
        except Exception as e:
            logger.warning("Failed to parse questions JSON: %s", e)
            parsed = None

        if isinstance(parsed, list):
            q_score, q_raw, q_fb = validate_questions_structure(parsed)
            logger.info("Questions structure score=%.3f raw=%s fb=%s", q_score, q_raw, q_fb)
            if q_score >= threshold:
                return parsed
            else:
                prompt += "\n\nFEEDBACK: " + "; ".join(q_fb)
        else:
            logger.warning("Questions generation not JSON list on attempt %d", attempt)

        time.sleep(4)

    logger.warning("Failed to generate valid questions after %d attempts; returning best-effort parse", max_attempts)
    return parsed if parsed is not None else []


def distractors_executor(prompt_template: str, topic: str, outputs_so_far: Dict[str, Any]) -> List[Dict]:
    """
    Generate distractors aligned with KG patterns for the passage.
    Returns: list of distractor objects.
    """
    passage = outputs_so_far.get("passage")
    if not passage:
        raise ValueError("distractors_executor requires 'passage' in outputs_so_far")

    d_rules = MCP.get_distractor_patterns()
    d_rules_txt = json.dumps(d_rules, ensure_ascii=False, indent=2)

    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        logger.info("DISTRACTORS_GENERATE attempt %d/%d", attempt, max_attempts)
        prompt = _DISTRACTOR_TEMPLATE.format(passage=passage, d_rules=d_rules_txt)
        if DEBUG_STUB:
            model_out = json.dumps([{"for_question_id":"Q1","distractors":[{"text":"DEBUG wrong","pattern":"similar_lexical"}]}])
        else:
            model_out = call_llm(prompt, expect_json=False, max_tokens=10000)

        try:
            parsed = _parse_json_from_model(model_out)
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            logger.warning("Failed to parse distractors JSON: %s", e)

        time.sleep(0.3)

    logger.warning("Returning last distractors attempt result (may be invalid).")
    try:
        return parsed if parsed is not None else []
    except UnboundLocalError:
        return []
