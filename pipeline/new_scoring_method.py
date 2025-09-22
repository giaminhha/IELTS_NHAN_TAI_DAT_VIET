
# ----------------------------
# gepa.py - GEPA driver (multi-objective, Pareto-aware) - UPGRADED
import random, os, string
import uuid
from typing import List, Dict, Callable, Any
import json, os, math, time
from pipeline.executors import passage_executor, questions_executor, validate_passage_text
from pipeline.validators import score_passages_only
from config import LLM_API_KEY, OPENAI_BASE_URL
from openai import OpenAI


def rewrite_based_on(passage:str, feedbacks : str):
    EVAL_PROMPT = f"""
    You are an IELTS Reading Passage Rewriter.  
    Your task is to take an existing passage and improve it so it better resembles an authentic IELTS Reading passage.

    ### Instructions:
    - Keep the **academic but accessible style** used in IELTS.
    - Revise according to the provided feedback categories.
    - Ensure the passage is about **850–950 words**, divided into **8 labeled paragraphs (A–H)**.
    - Maintain a **balance of sentence types**:
      - About 60% complex sentences
      - About 40% simple/compound sentences
      - Max sentence length: 35 words
      - Most sentence should be around 25 - 30 words, or less. Add more simple sentence so that the passage can be familiar with 6.0 - 7.0 students.
    - Vocabulary:
      - Target CEFR B2–C1
      - Academic Word List (AWL) should appear frequently
      - Limit rare/technical words to <5%, and define them if used
    - Readability:
      - Flesch Reading Ease: 40–60
      - Flesch–Kincaid Grade: 9–11
    - Content:
      - Mix of factual explanation, examples, statistics, and one case study
      - Neutral, informative tone (not persuasive or emotional)
      - Finish with a **1–2 sentence summary line** starting with "Summary: ..."

    ### Input:
    Original Passage:
    {passage}

    Feedbacks:
    {feedbacks}

    ### Output:
    A fully rewritten IELTS-like passage that adjusted based on all feedbacks, with paragraphs A–H and a final Summary line.
    """

    client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE_URL)
    # attempts + validation loop
    max_attempts = 3
    threshold = 0.70  # passage validator score threshold
    last_traces = None

    final_passage = ""
    for attempt in range(1, max_attempts + 1):
        print("PASSAGE_REGENERATE attempt %d/%d", attempt, max_attempts)

        prompt = EVAL_PROMPT
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "developer", "content": "You are an IELTS Reading examiner assistant."},
                    {"role": "user", "content": EVAL_PROMPT}],
            temperature=0.7
        )
        # Extract the generated passage text
        final_passage = response.choices[0].message.content.strip()

        # validate
        score, raw_traces, fb_traces = validate_passage_text(final_passage)
        print("Passage score=%.3f raw=%s feedback=%s", score, raw_traces, fb_traces)
        last_traces = raw_traces

        if score >= threshold:
            return passage
    
            # small wait/backoff to avoid rate-limit spikes
        time.sleep(5)
    print(f"PASSAGE generation failed to meet threshold after {max_attempts} attempts. Returning last result.")
    return final_passage
def generate_passage_with_rescoring(
    executors: Dict[str, Callable[[str, str, dict], Any]],
    base_prompts: Dict[str, str],
    topic: str,
    threshold: float = 0.8,
    max_attempts: int = 8):
    """
    Generate a passage using executors.passage_executor, then rescore using score_passages_only.
    If any score < threshold, use fb_traces as feedback to regenerate until all scores >= threshold.
    Returns: (final_passage, final_scores, final_traces)
    """
      # adjust import path if needed

    outputs = {}
    attempt = 0
    fb_prompt_suffix = ""

    passage = passage_executor(prompt_template="", topic=topic, outputs_so_far=outputs)
    while passage == "Failure Passage":
        time.sleep(5)
        passage = passage_executor(prompt_template="", topic=topic, outputs_so_far=outputs)
    while attempt < max_attempts:
        attempt += 1
        # ----- Generate passage -----
        if attempt > 1:
            passage = rewrite_based_on(passage, outputs["feedback"])

        # Store so we can pass to scorers if needed
        outputs["passage"] = passage

        # ----- Score passage -----
        scores, traces_dict = score_passages_only(outputs, topic)
        displayed_scores = {k: float(v) for k, v in scores.items()}
        traces = traces_dict.get("raw", [])
        fb_traces = traces_dict.get("feedback", [])
        print(f"[RESCORE] Attempt {attempt} scores={displayed_scores}")
        print(f"{passage}")
        # Check if all scores >= threshold
        if all(v >= threshold for v in scores.values()):
            gen_questions(topic, outputs)
            return

        # Otherwise: build feedback and retry
        feedback_text = "; ".join(fb_traces) if fb_traces else "improve clarity and alignment with IELTS style"
        fb_prompt_suffix = f"\n\nFEEDBACK from evaluator: {feedback_text}"

        # Inject feedback into outputs for next executor call
        outputs["feedback"] = fb_prompt_suffix
        time.sleep(5)

    print(f"[RESCORE] Failed to reach threshold after {max_attempts} attempts, returning last result.")
    gen_questions(topic, outputs)


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
            qtype = q.get("question_type", "").strip()
            if qtype == "True/False/Not Given" or qtype == "MCQ":
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
            elif qtype == "Matching Headings":
                # Instruction line
                lines.append("Text: " + q["question_text"])
                lines.append("")  # blank line
                
                # Headings list
                for heading in q["headings"]:
                    lines.append("Text: " + heading)
                lines.append("")  # blank line
                
                # Paragraph-answer-rationale blocks
                for i, para in enumerate(q["paragraphs"], start=1):
                    ans = q["answers"].get(para, "")
                    ans_prefix = ans.split('.')[0].strip() if ans else ""
                    rationale_text = q.get("rationale", {}).get(para, "")
                    
                    lines.append(f"{i}. Paragraph {para}")
                    if rationale_text:
                        lines.append(f"... {rationale_text}")
                    lines.append(f"* {ans_prefix}")  
                    lines.append("")  # blank line
            elif qtype == "Sentence Completion":
                for q in questions:
                    # Instruction line
                    lines.append(f"Text: Questions for {q['id']}: Complete the passage below. "
                                "Choose NO MORE THAN TWO WORDS from the passage for each blank.")
                    lines.append("")

                    # Passage with blanks
                    lines.append("Text: " + q["Question_passage"])
                    lines.append("")

                    # Answers and rationales (sorted by blank number)
                    for key in sorted(q["answers"].keys(), key=lambda x: int(x)):
                        ans = q["answers"][key]
                        rationale = q["rationale"].get(key, "")
                        lines.append(f"{key}. Text: ... {rationale}")
                        lines.append(f"* {ans}")
                        lines.append("")
                # Short Answer (Answer the question in X words)
            elif qtype == "Short Answer":
                qtext = q.get("question_text", "").strip()
                answer = q.get("answer", "").strip()
                rationale = q.get("rationale", "").strip()

                lines.append(f"Text: {i}. {qtext}")
                if rationale:
                    lines.append(f"Text: ... {rationale}")
                lines.append(f"* {answer}")
            elif qtype == "Matching Features":
                # Header
                lines.append(f"Text: Questions {q['id'][1:]}–{int(q['id'][1:]) + len(q['statements']) - 1}")
                lines.append(f"Text: Look at the following statements (Questions {q['id'][1:]}–{int(q['id'][1:]) + len(q['statements']) - 1}) and the list of people below.")
                lines.append(f"Text: {q['question_text']}, A, B, or C.")
                lines.append(f"Text: Write the correct letter, A, B, or C, in boxes {q['id'][1:]}–{int(q['id'][1:]) + len(q['statements']) - 1} on your answer sheet.")
                lines.append("Text: NB You may use any letter more than once.")

                # Options / list of people
                lines.append("Text: List of People")
                for key, person in q.get('options', {}).items():
                    lines.append(f"Text: {key} {person}")

                # Statements
                for idx, statement in enumerate(q['statements'], start=int(q['id'][1:])):
                    lines.append(f"{idx}. {statement}")
                    rationale = q.get('rationale', {}).get(statement, "").strip()
                    if rationale:
                        lines.append(f"... {rationale}")
                    answer = q.get('answers', {}).get(statement, "").strip()
                    lines.append(f"* {answer}")
    # Write out
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[SAVED] {filename}")

def gen_questions(topic: str, outputs: Dict[str, Any]):
    out = questions_executor(prompt_template="", topic=topic, outputs_so_far=outputs)
    outputs["questions"] = out
    save_generation_passages(topic, outputs=outputs)