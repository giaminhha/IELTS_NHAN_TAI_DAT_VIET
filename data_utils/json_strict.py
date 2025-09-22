# ---------- safe_json_loads utility ----------
import json, re

# try to use json5 if available (more forgiving)
try:
    import json5
    _HAVE_JSON5 = True
except Exception:
    _HAVE_JSON5 = False


def _extract_first_braced(text: str):
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    return m.group(1) if m else None


def _cleanup_common_errors(s: str) -> str:
    # 1) strip surrounding commentary & extract first {...} or [...]
    cand = _extract_first_braced(s) or s

    # 2) remove trailing commas before } or ]
    cand = re.sub(r',\s*([}\]])', r'\1', cand)

    # 3) fix missing commas between string/number and next key
    # e.g.  "answer": "A"  "options":  ->  "answer": "A", "options":
    cand = re.sub(r'(".*?"|\d)(\s*")', r'\1,\2', cand)

    # 4) fix missing commas between object/array and next key
    cand = re.sub(r'([}\]])(\s*")', r'\1,\2', cand)

    # 5) optionally fix lone single-quotes -> double-quotes (simple heuristic)
    if '"' not in cand and "'" in cand:
        cand = cand.replace("'", '"')

    return cand


def safe_json_loads(text: str):
    """
    Try strict json.loads; if it fails, try json5 (if available),
    then try to extract the first {...} / [...] and run a light cleanup and parse.
    Returns parsed object on success; on failure returns a dict with parse info:
        {"__raw": original_text, "__parse_error": "<error msg>"}
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    if not s:
        return {"__raw": text, "__parse_error": "empty string"}

    e_strict = None  # ensure always defined

    # 1) strict JSON
    try:
        return json.loads(s)
    except Exception as e:
        e_strict = e  # store the error

    # 2) try json5 if installed
    if _HAVE_JSON5:
        try:
            return json5.loads(s)
        except Exception:
            pass

    # 3) try extraction + cleanup
    cleaned = _cleanup_common_errors(s)
    try:
        return json.loads(cleaned)
    except Exception as e_clean:
        return {"__raw": s, "__parse_error": f"strict:{e_strict} | clean:{e_clean}"}
