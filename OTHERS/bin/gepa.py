
# ----------------------------
# gepa.py - GEPA driver (multi-objective, Pareto-aware) - UPGRADED
import random
import uuid
from copy import deepcopy
from typing import List, Dict, Callable, Any
from statistics import mean
import json, os, math, time
from llm import call_llm
from validators import score_passage_and_questions
from logger import GEPA_Logger

# ---------- CONFIG ----------
MINIBATCH_SIZE = 8   #8 later on # small for sample efficiency
NPARETO = 20 # 20 - 30% amount of topics
MAX_CANDIDATES = 40 #30 - 50
ROLLOUT_BUDGET = 700  #scale to 500 - 1000, final research runs 2000+ if have credits
MUTATION_ATTEMPTS = 2 # 2 - 3 if more variety
INIT_POPULATION = 12      # new: initial diverse candidates
ELITISM_RATIO = 0.15     # keep top 15% each generation
DOMINANCE_EPS = 0.02     # epsilon tolerance for dominance
MIN_POOL_SIZE = 3        # maintain at least this many candidates
# ----------------------------

"""
🔹 Full Research / Convergence Run

Use when you want to actually evolve strong prompts (credits heavy).

MINIBATCH_SIZE = 12
NPARETO = 30
MAX_CANDIDATES = 60
ROLLOUT_BUDGET = 2000
MUTATION_ATTEMPTS = 3
INIT_POPULATION = 20
ELITISM_RATIO = 0.20
DOMINANCE_EPS = 0.01
MIN_POOL_SIZE = 5
"""
MODULE_NAMES = ["passage", "questions"]


import os
import string

def save_generation_passages(topic: str, outputs: dict):
    # Sanitize topic name for filename
    safe_topic = topic.replace(" ", "_")

    # Count existing generations under "tests/"
    base_dir = "tests"
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.isdigit()]
    gen_num = len(existing) + 1

    # Folder like "tests/12"
    folder = os.path.join(base_dir, str(gen_num))
    os.makedirs(folder, exist_ok=True)

    # Save file as "tests/12/<topic>.txt"
    filename = os.path.join(folder, f"{safe_topic}.txt")

    passage = outputs.get("passage", "").strip()
    questions = outputs.get("questions", [])

    lines = []
    # --- Passage ---
    lines.append(f"Text title: {topic}\n")
    lines.append(f"Text: {passage}\n")

    # --- Questions ---
    if questions:
        lines.append("Text: Questions\n")
        for i, q in enumerate(questions, 1):
            qtext = q.get("question_text", "").strip()
            rationale = q.get("rationale", "").strip()
            options = q.get("options", [])
            answer = q.get("answer", "").strip()

            lines.append(f"{i}. {qtext}")
            if rationale:
                lines.append(f"... {rationale}")

            for j, opt in enumerate(options):
                label = string.ascii_lowercase[j]
                if opt.strip() == answer:
                    lines.append(f"*{label}) {opt}")
                else:
                    lines.append(f"{label}) {opt}")
            lines.append("")

    # Write out
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[SAVED] {filename}")



# ---------- Utilities ----------
def new_candidate_from_base(base_prompts: Dict[str, str], extra_meta=None) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "prompts": deepcopy(base_prompts),
        "scores": {},      # last aggregated scores (dict)
        "meta": extra_meta or {},
        "ancestry": [],
        "traces": [],      # raw traces from last eval
        "fb_traces": [],   # feedback-style traces from last eval
    }


# ---------- Rollout / evaluation ----------
def run_rollout_on_topic(candidate: Dict, topic: str,
                         executors: Dict[str, Callable[[str, str, dict], Any]],
                         verbose: bool = False) -> Dict:
    outputs: Dict[str, Any] = {}
    for m in MODULE_NAMES:
        prompt_text = candidate["prompts"].get(m)
        if prompt_text is None:
            raise RuntimeError(f"Candidate missing prompt for module {m}")

        out = executors[m](prompt_text, topic, outputs)
        outputs[m] = out

    # validators.score_passage_and_questions returns (scores, traces_dict)
    scores, traces_dict = score_passage_and_questions(outputs, topic)
    candidate["scores"] = {k: float(v) for k, v in scores.items()}
    candidate["traces"] = traces_dict.get("raw", [])
    candidate["fb_traces"] = traces_dict.get("feedback", [])

    # persist generation (useful for later analysis)
    # try:
    #    save_generation_passages(topic, outputs)
    # except Exception as e:
    #    print("[WARN] failed saving generation outputs:", e)

    return {"topic": topic, "outputs": outputs, "scores": candidate["scores"],
            "raw_traces": candidate["traces"], "fb_traces": candidate["fb_traces"]}


def run_minibatch(candidate: Dict, topics: List[str],
                  executors: Dict[str, Callable[[str, str, dict], Any]]) -> List[Dict]:
    results = []
    for t in topics:
        res = run_rollout_on_topic(candidate, t, executors)
        results.append(res)
    return results


# ---------- Multi-objective helpers ----------
def dominates(a: Dict[str, float], b: Dict[str, float], eps: float = DOMINANCE_EPS) -> bool:
    """
    A dominates B with tolerance eps.
    """
    keys = set(a.keys()) | set(b.keys())
    ge_all = all(a.get(k, 0.0) >= b.get(k, 0.0) - eps for k in keys)
    gt_some = any(a.get(k, 0.0) > b.get(k, 0.0) + eps for k in keys)
    return ge_all and gt_some


def aggregate_scores(results: List[Dict]) -> Dict[str, float]:
    """
    results: list of rollout dicts (each has 'scores' dict)
    returns mean per objective
    """
    if not results:
        return {}
    keys = set()
    for r in results:
        if isinstance(r.get("scores"), dict):
            keys.update(r["scores"].keys())
    agg = {}
    for k in keys:
        vals = [r.get("scores", {}).get(k, 0.0) for r in results]
        agg[k] = mean(vals) if vals else 0.0
    return agg


def build_pareto_front(records: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    """
    records: { candidate_id: { topic: scores_dict, ... }, ... }
    Compute avg vector per candidate then non-dominated set.
    """
    avg_vectors: Dict[str, Dict[str, float]] = {}
    for cid, topic_map in records.items():
        topic_results = list(topic_map.values())
        if not topic_results:
            avg_vectors[cid] = {}
            continue
        # each topic_result is a scores dict
        normed = [{"scores": s if isinstance(s, dict) else {"_single": float(s)}} for s in topic_results]
        avg_vectors[cid] = aggregate_scores(normed)

    pareto = []
    for a in avg_vectors:
        a_vec = avg_vectors[a]
        dominated = False
        for b in avg_vectors:
            if a == b:
                continue
            b_vec = avg_vectors[b]
            if dominates(b_vec, a_vec):
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    return pareto


# ---------- GEPA meta-prompt ----------
GEPA_META_PROMPT_TEMPLATE = """
I provided an assistant with the following instruction (the module prompt) delimited by triple quotes:

'''
{current_instruction}
'''

Here are a few examples of inputs, outputs, and feedback from runs:

{examples_text}

Your task: write a new improved instruction for the assistant (the same module) that
- fixes the problems called out in the feedback,
- includes any domain-specific constraints implied by the examples,
- is explicit and repeatable,
- keep it concise.

Return ONLY the new instruction inside triple backticks.
"""


def make_meta_prompt(current_instruction: str, examples: List[Dict]) -> str:
    ex_texts = []
    for ex in examples:
        ex_texts.append(f"Input: {ex['input']}\nOutput: {ex['output']}\nFeedback: {ex['feedback']}\n---")
    return GEPA_META_PROMPT_TEMPLATE.format(current_instruction=current_instruction,
                                            examples_text="\n".join(ex_texts))


def reflective_prompt_mutation(module_name: str, current_prompt: str, examples: List[Dict]) -> str:
    """Call LLM meta-prompt; fallback to a light synthetic mutation if LLM returns nothing useful."""
    meta = make_meta_prompt(current_prompt, examples)
    try:
        resp = call_llm(meta)
    except Exception as e:
        print("[WARN] meta LLM mutation failed:", e)
        resp = ""
    new_instr = resp.strip() if isinstance(resp, str) else ""
    if new_instr.startswith("```") and new_instr.endswith("```"):
        new_instr = new_instr.strip("`").strip()
    # fallback simple mutation if empty
    if not new_instr or len(new_instr) < 20:
        # heuristic feedback-driven small edits
        fb_concat = "; ".join([ex.get("feedback", "") for ex in examples])[:800]
        if "short" in fb_concat.lower():
            new_instr = current_prompt + "\n(Ensure passage length ~900 words; expand paragraphs.)"
        elif "missing summary" in fb_concat.lower():
            new_instr = current_prompt + "\n(Include final line: 'Summary: <one-line summary>')"
        else:
            # random minor mutation
            random
            new_instr = current_prompt + "\n(Add more formal tone; add 2 academic words per paragraph.)"
    return new_instr


def system_merge_prompts(prompt_a: str, prompt_b: str) -> str:
    lines = []
    for s in (prompt_a + "\n" + prompt_b).splitlines():
        s_clean = s.strip()
        if not s_clean:
            continue
        if s_clean not in lines:
            lines.append(s_clean)
    return "\n".join(lines)


# ---------- Candidate pool ----------
class CandidatePool:
    def __init__(self, base_prompts: Dict[str, str], max_size=MAX_CANDIDATES):
        self.base_prompts = deepcopy(base_prompts)
        self.pool: Dict[str, Dict] = {}
        self.max_size = max_size
        # initialize with a small diverse population
        for i in range(INIT_POPULATION):
            c = new_candidate_from_base(base_prompts, extra_meta={"seed": f"init_{i}"})
            # apply tiny synthetic variations
            if i % 3 == 0:
                c["prompts"] = {k: v + "\nEnsure ~900 words." for k, v in c["prompts"].items()}
            elif i % 3 == 1:
                c["prompts"] = {k: v + "\nUse formal academic tone; avoid contractions." for k, v in c["prompts"].items()}
            else:
                c["prompts"] = {k: v + "\nProvide inline citations for factual claims." for k, v in c["prompts"].items()}
            self.add_candidate(c)

    def add_candidate(self, cand: Dict):
        self.pool[cand["id"]] = cand
        self._trim_pool()

    def remove_candidate(self, cid: str):
        if cid in self.pool:
            del self.pool[cid]

    def _trim_pool(self):
        if len(self.pool) > self.max_size:
            # remove lowest-scoring (approx) if scores available, else random
            try:
                scored = [(cid, cand.get("scores", {}).get("passage", 0.0)) for cid, cand in self.pool.items()]
                scored_sorted = sorted(scored, key=lambda x: x[1])
                while len(self.pool) > self.max_size:
                    rem = scored_sorted.pop(0)[0]
                    if rem in self.pool:
                        del self.pool[rem]
            except Exception:
                while len(self.pool) > self.max_size:
                    key = random.choice(list(self.pool.keys()))
                    del self.pool[key]

    def list_candidates(self) -> List[Dict]:
        return list(self.pool.values())

def get_best_candidate(pool: "CandidatePool") -> Dict[str, Any]:
    """
    Select the best candidate from the pool based on average of its scores.
    Returns the candidate dict.
    """
    best_cand = None
    best_score = float("-inf")

    for cand in pool.list_candidates():
        scores = cand.get("scores", {})
        if not scores:
            continue
        avg_score = sum(scores.values()) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_cand = cand

    if best_cand is None:
        print("[WARN] No candidate had scores; returning a random one.")
        return random.choice(pool.list_candidates()) if pool.list_candidates() else None

    print(f"[GEPA] Best candidate {best_cand['id']} with avg_score {best_score:.2f}")
    return best_cand

# ---------- GEPA main loop ----------
def gepa_optimize(
    executors: Dict[str, Callable[[str, str, dict], Any]],
    base_prompts: Dict[str, str],
    topics: List[str],
    dpareto_size: int = NPARETO,
    budget: int = ROLLOUT_BUDGET
):
    pool = CandidatePool(base_prompts)
    random.shuffle(topics)
    dfeedback = topics[:min(len(topics), 200)]
    dpareto = topics[:min(len(topics), dpareto_size)]
    logger = GEPA_Logger()
    rollouts_used = 0
    iteration = 0
    # records: candidate_id -> topic -> scores_dict
    records: Dict[str, Dict[str, Dict[str, float]]] = {}
    for c in pool.list_candidates():
        records[c["id"]] = {}

    # auxiliary helper: ensure minimal population diversity
    def ensure_min_pool(n=MIN_POOL_SIZE):
        while len(pool.pool) < n:
            # clone a random existing candidate and slightly mutate
            if pool.pool:
                parent = random.choice(list(pool.pool.values()))
                child = deepcopy(parent)
                child["id"] = str(uuid.uuid4())
                # small synthetic change
                for m in child["prompts"]:
                    child["prompts"][m] = child["prompts"][m] + "\n(Variation: add connective markers.)"
                pool.add_candidate(child)
                records[child["id"]] = {}
            else:
                # create fresh from base
                pool.add_candidate(new_candidate_from_base(base_prompts))
                records[list(pool.pool.keys())[-1]] = {}

    ensure_min_pool()

    while rollouts_used < budget:
        iteration += 1
        print(f"[GEPA] iter {iteration}, rollouts_used {rollouts_used}, pool_size {len(pool.pool)}")

        candidates = pool.list_candidates()
        parent = random.choice(candidates)
        module_to_mutate = random.choice(MODULE_NAMES)
        minibatch = random.sample(dfeedback, k=min(MINIBATCH_SIZE, len(dfeedback)))

        # evaluate parent on minibatch
        parent_results = run_minibatch(parent, minibatch, executors)
        rollouts_used += len(parent_results)
        rec = records.setdefault(parent["id"], {})
        for r in parent_results:
            rec[r["topic"]] = r["scores"]

        # prepare examples for meta-prompt (feedback-only short)
        examples = []
        for r in parent_results:
            passage = r["outputs"].get("passage", "")
            out_summary = (passage[:200].replace("\n", " ")) if isinstance(passage, str) else str(passage)
            fb_text = "; ".join(r.get("fb_traces", []))  # only feedback traces for mutation
            examples.append({"input": r["topic"], "output": out_summary, "feedback": fb_text})

        # attempt mutation (use LLM reflective mutation, with MUTATION_ATTEMPTS)
        new_prompts: Dict[str, str] = {}
        for attempt in range(MUTATION_ATTEMPTS):
            current_instr = parent["prompts"][module_to_mutate]
            mutated = reflective_prompt_mutation(module_to_mutate, current_instr, examples)
            if mutated and len(mutated) > 20 and mutated != current_instr:
                new_prompts[module_to_mutate] = mutated
                break

        if not new_prompts:
            # fallback: synthetic local mutation using feedback
            fb_concat = "; ".join([ex["feedback"] for ex in examples])[:500]
            fallback = parent["prompts"][module_to_mutate]
            if "short" in fb_concat.lower():
                fallback = fallback + "\n(Please expand passage: target ~900 words.)"
            elif "missing summary" in fb_concat.lower():
                fallback = fallback + "\n(Include Summary: at the end.)"
            else:
                fallback = fallback + "\n(Add cohesion markers: however, moreover; add 2 academic words per paragraph.)"
            new_prompts[module_to_mutate] = fallback

        # build child candidate
        child = deepcopy(parent)
        child["id"] = str(uuid.uuid4())
        child["prompts"] = deepcopy(parent["prompts"])
        child["prompts"].update(new_prompts)
        child["ancestry"] = parent.get("ancestry", []) + [parent["id"]]

        # evaluate child on minibatch
        child_results = run_minibatch(child, minibatch, executors)
        rollouts_used += len(child_results)

        # aggregate per-objective means across minibatch
        parent_vec = aggregate_scores(parent_results)
        child_vec = aggregate_scores(child_results)
        print(f"[GEPA] parent_vec {parent_vec}")
        print(f"[GEPA] child_vec  {child_vec}")

        # require Pareto dominance on minibatch to consider extended evaluation
        if not dominates(child_vec, parent_vec):
            print("[GEPA] Child did not dominate parent on minibatch; attempting occasional acceptance for diversity.")
            # occasional exploratory acceptance (small probability) to keep diversity
            if random.random() < 0.10:
                print("[GEPA] Accepting child by exploration policy.")
                pool.add_candidate(child)
                records[child["id"]] = {r["topic"]: r["scores"] for r in child_results}
            else:
                # reject
                continue
        else:
            # extended Pareto evaluation
            pareto_eval_topics = random.sample(dpareto, k=min(len(dpareto), max(4, dpareto_size // 5)))
            parent_pareto = run_minibatch(parent, pareto_eval_topics, executors)
            child_pareto = run_minibatch(child, pareto_eval_topics, executors)
            rollouts_used += len(pareto_eval_topics) * 2

            parent_p_vec = aggregate_scores(parent_pareto)
            child_p_vec = aggregate_scores(child_pareto)
            print(f"[GEPA] parent_p_vec {parent_p_vec}")
            print(f"[GEPA] child_p_vec  {child_p_vec}")

            if dominates(child_p_vec, parent_p_vec):
                print(f"[GEPA] Accepting child {child['id']}.")
                pool.add_candidate(child)
                records[child["id"]] = {r["topic"]: r["scores"] for r in child_pareto}
                records[parent["id"]] = {r["topic"]: r["scores"] for r in parent_pareto}
            else:
                print("[GEPA] Child failed on pareto set; rejecting.")

        # keep minimum pool size so search doesn't collapse
        ensure_min_pool()

        if iteration % 10 == 0:
            pareto_ids = build_pareto_front(records)
            print(f"[GEPA] pareto front size: {len(pareto_ids)}")
        logger.log({
            "iteration": iteration,
            "parent_id": parent["id"],
            "child_id": child["id"],
            "parent_scores": parent.get("scores", {}),
            "child_scores": child.get("scores", {}),
            "examples": examples,
            "traces": child.get("traces", [])
        })

    print("[GEPA] Budget exhausted.")
    
    # --- extract final best candidate ---
    best = get_best_candidate(pool)

    # Pick a representative topic (e.g. the first topic in your list)
    final_topic = topics[0]

    # Re-run the best candidate to get its passage + questions
    final_outputs = run_rollout_on_topic(best, final_topic, executors)

    # Save using your existing function
    save_generation_passages(final_topic, final_outputs["outputs"])

    return best
