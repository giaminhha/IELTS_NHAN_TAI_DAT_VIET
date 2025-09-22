# retriever.py
import requests
import re
from googlesearch import search   # your existing web-search tool
from pipeline.llm import call_llm
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"


def fetch_from_semantic_scholar(topic, limit=3):
    params = {
        "query": topic,
        "limit": limit,
        "fields": "title,abstract,year,url"
    }
    try:
        resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("Semantic Scholar fetch failed:", e)
        return []

    results = []
    for i, paper in enumerate(data.get("data", []), 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year")
        abstract = paper.get("abstract") or title
        url = paper.get("url", "")
        facts = []
        if year:
            facts.append(f"Year: {year}")
        if url:
            facts.append(f"Source: {url}")
        results.append(normalize_source("S", i, title, abstract=abstract, facts=facts, url=url))
    return results


def fetch_from_openalex(topic, limit=3):
    params = {"search": topic, "per-page": limit}
    try:
        resp = requests.get(OPENALEX_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("OpenAlex fetch failed:", e)
        return []

    results = []
    for i, work in enumerate(data.get("results", []), 1):
        title = work.get("title", "Untitled")
        year = work.get("publication_year", "Unknown")
        doi = work.get("doi", "")
        abstract = work.get("abstract", "") or title
        facts = []
        if year and year != "Unknown":
            facts.append(f"Year: {year}")
        if doi:
            facts.append(f"DOI: {doi}")
        results.append(normalize_source("O", i, title, abstract=abstract, facts=facts))
    return results


def fetch_from_crossref(topic, limit=3):
    params = {"query": topic, "rows": limit}
    try:
        resp = requests.get(CROSSREF_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("CrossRef fetch failed:", e)
        return []

    results = []
    for i, item in enumerate(data.get("message", {}).get("items", []), 1):
        title = " ".join(item.get("title", [])) or "Untitled"
        year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        doi = item.get("DOI", "")
        facts = []
        if year:
            facts.append(f"Year: {year}")
        if doi:
            facts.append(f"DOI: {doi}")
        results.append(normalize_source("C", i, title, abstract=title, facts=facts))
    return results

def fetch_from_web(topic, limit=3):
    try:
        results = list(search(f"{topic} research facts experiments events"))  # convert to list
    except Exception as e:
        print("Web search failed:", e)
        return []

    sources = []
    for i, r in enumerate(results[:limit], 1):
        # depending on your googlesearch version, r might be a string (URL) not dict
        if isinstance(r, str):
            title = r
            snippet = ""
            url = r
        else:
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "")
            url = r.get("url", "")

        abstract = snippet or title
        facts = []
        if url:
            facts.append(f"Source: {url}")
        sources.append(normalize_source("W", i, title, abstract=abstract, facts=facts, url=url))
    return sources


def highlight_facts(text):
    """
    Extract and emphasize numbers, years, percentages, and sample sizes 
    so the LLM uses them more reliably in passages.
    """
    # Bold numbers and years in brackets (so they stand out in the prompt)
    text = re.sub(r"(\b\d{4}\b)", r"<YEAR:\1>", text)        # years
    text = re.sub(r"(\b\d+%|\b\d+\.\d+%|\b\d+\b)", r"<NUM:\1>", text)  # numbers/percentages
    return text


def process_sources(sources: list[dict]) -> list[dict]:
    """
    Apply highlighting to abstracts of all sources.
    """
    processed = []
    for s in sources:
        s_copy = dict(s)  # shallow copy
        if "abstract" in s_copy and s_copy["abstract"]:
            s_copy["abstract"] = highlight_facts(s_copy["abstract"])
        processed.append(s_copy)
    return processed


def retrieve_sources(topic: str, limit: int = 5):
    """
    Use LLM to suggest academic-style sources for a topic.
    Returns a list of dicts with id, abstract, facts.
    """
    prompt = f"""
    You are an assistant that finds academic references.

    Task: List {limit} academic papers about the topic "{topic}".
    For each paper, return:
      - title
      - year
      - abstract (1–3 sentences)
      - optional URL if known

    Output format: JSON array of objects with keys:
      "title", "year", "abstract", "url"
    """

    resp = call_llm(prompt, expect_json=True)

    sources = []
    if isinstance(resp, list):
        for i, item in enumerate(resp[:limit], 1):
            title = item.get("title", "Untitled")
            year = item.get("year", "")
            abstract = item.get("abstract", title)
            url = item.get("url", "")

            facts = []
            if year:
                facts.append(f"Year: {year}")
            if url:
                facts.append(f"Source: {url}")

            sources.append({
                "id": f"L{i}",
                "abstract": abstract,
                "facts": facts
            })
    else:
        # fallback: treat raw text as one abstract
        sources = [{
            "id": "L1",
            "abstract": str(resp),
            "facts": []
        }]

    sources = deduplicate_sources(sources)
    return sources


def normalize_source(id_prefix: str, idx: int, title: str,
                     abstract: str | None = None,
                     facts: list[str] | None = None,
                     url: str | None = None) -> dict:
    """
    Normalize source into {id, abstract, facts}.
    - abstract: main text body
    - facts: list of short factual strings (years, %s, DOIs, URLs, etc.)
    """
    facts = facts or []
    if url:
        facts.append(f"Source: {url}")
    return {
        "id": f"{id_prefix}{idx}",
        "abstract": abstract or title,
        "facts": facts
    }


def deduplicate_sources(sources: list[dict]) -> list[dict]:
    """
    Deduplicate sources by abstract text (case-insensitive).
    """
    seen = set()
    unique = []
    for s in sources:
        key = s.get("abstract") or s.get("text", "")
        key = key.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

