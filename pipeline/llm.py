# llm.py
"""
Robust LLM wrapper for your pipeline.

Features:
- DEBUG_STUB mode (controlled via env DEBUG_STUB)
- Caching   
- Safe JSON extraction when expect_json=True
- Graceful handling of missing config / SDK differences
"""

import os
import json
import re
import hashlib
from typing import Optional, Any, List
from data_utils.json_strict import safe_json_loads
# Try to import OpenAI client; give friendly error if missing
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    _openai_import_error = e

from config import LLM_API_KEY as _CFG_KEY, MODEL as _CFG_MODEL, OPENAI_BASE_URL as _CFG_BASE

LLM_API_KEY = _CFG_KEY
MODEL = _CFG_MODEL
OPENAI_BASE = _CFG_BASE

# DEBUG stub mode
DEBUG_STUB = False

# If not in debug and OpenAI client missing, warn early
if not DEBUG_STUB and OpenAI is None:
    raise ImportError(
        f"OpenAI client import failed: {_openai_import_error}\n"
        "Install the official `openai` Python package (pip install openai) or set DEBUG_STUB=true for local testing."
    )

# Create client lazily (so DEBUG_STUB can run without keys)
_client: Optional[Any] = None
def _init_client():
    global _client
    if _client is not None:
        return _client
    if DEBUG_STUB:
        return None
    # require API key
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not set (env or config).")
    # instantiate OpenAI/OpenRouter client
    try:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE)
    except TypeError:
        # Some OpenAI client versions have different param names
        _client = OpenAI(api_key=LLM_API_KEY)
    return _client

# Simple in-memory cache
_llm_cache = {}

def _cache_key(prompt: str, system: Optional[str]):
    key = (system or "") + "\n" + prompt
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _stub_passage(topic_snippet: str):
    text = f"{topic_snippet} is an important contemporary topic. " * 30
    summary = "Summary: This passage briefly discusses the topic."
    return text + "\n\n" + summary

def _extract_json_from_text(text: str):
    """
    Try to find and parse the first JSON object/array in `text`.
    Returns parsed object or raises ValueError.
    """
    # look for top-level JSON array or object
    m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.S)
    if not m:
        raise ValueError("No JSON object/array found in LLM output.")
    candidate = m.group(1)
    parsed = safe_json_loads(candidate)
    if isinstance(parsed, dict) and parsed.get("__parse_error"):
        # keep raising so upstream can handle/record parse_error
        raise ValueError("No valid JSON object found in candidate: " + parsed["__parse_error"])
    return parsed

def _extract_text_from_completion(completion):
    """
    Robust extraction of assistant text from various SDK return shapes.
    Supports:
      - completion.choices[0].message.content
      - completion.choices[0].message (dict)
      - completion.choices[0].text
      - string (already)
    """
    try:
        # OpenAI-style: completion.choices[0].message.content
        choice = completion.choices[0]
        if hasattr(choice, "message"):
            msg = choice.message
            # if msg has .content attribute
            content = getattr(msg, "content", None)
            if content is not None:
                return content
            # if msg is dict-like
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
        # fallback to .text (older SDKs)
        text = getattr(choice, "text", None)
        if text:
            return text
    except Exception:
        pass

    # If completion itself is a string
    if isinstance(completion, str):
        return completion

    # Last resort: try to stringify
    try:
        return json.dumps(completion)
    except Exception:
        return str(completion)

def call_llm(prompt: str, system: Optional[str] = None,
             expect_json: bool = False, use_cache: bool = True,
             tools: Optional[List[dict]] = None, max_tokens: int = 10000) -> Any:
    """
    Provider-aware LLM call interface.
    Supports both OpenAI-style and Bedrock-style models.
    """
    model = MODEL
    key = _cache_key(prompt, system)
    if use_cache and key in _llm_cache:
        return _llm_cache[key]

    # DEBUG STUB (skip API call)
    if DEBUG_STUB:
        low = prompt.lower()
        if "ielts" in low and ("passage" in low or "academic reading" in low):
            out = _stub_passage(prompt[:80])
        elif "question" in low or expect_json:
            out = json.dumps([
                {"id": "MCQ_1", "question_text": "Stub: What is X?", "options": ["A","B","C","D"], "answer": "A", "rationale": "sample", "linked_skills": ["Inference"]},
                {"id": "MCQ_2", "question_text": "Stub: Which is true?", "options": ["A","B","C","D"], "answer": "B", "rationale": "sample", "linked_skills": ["Scanning"]}
            ])
        else:
            out = "FAKE OUTPUT: " + (prompt[:200] if len(prompt) > 200 else prompt)

        if use_cache:
            _llm_cache[key] = out
        if expect_json:
            parsed = safe_json_loads(out)
            # if parse failed, return raw (previous behavior) but preserve parse info
            if isinstance(parsed, dict) and parsed.get("__parse_error"):
                return {"raw": out, "parse_error": parsed["__parse_error"]}
            return parsed
        return out   # ✅ MUST return here


    client = _init_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        kwargs = {
            "model": model,
            "messages": messages,
        }
        model_lower = model.lower().strip()

        # Bedrock Anthropic models
        if model_lower.startswith("anthropic/") or model_lower.startswith("anthropic.") or "claude" in model_lower or model_lower.startswith("amazon."):
            # === Bedrock-style ===
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
        else:
            # === OpenAI-style ===
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

        completion = client.chat.completions.create(**kwargs)
        text = _extract_text_from_completion(completion)

        if use_cache:
            _llm_cache[key] = text

        if expect_json:
            parsed = safe_json_loads(text)
            if isinstance(parsed, dict) and parsed.get("__parse_error"):
                # fallback to explicit extractor (keeps previous behavior)
                try:
                    return _extract_json_from_text(text)
                except Exception as e:
                    return {"raw": text, "parse_error": parsed["__parse_error"] + " | " + str(e)}
            return parsed
        return text

    except Exception as e:
        raise RuntimeError(f"LLM call failed (model={model}): {e}")
