
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
from data_utils.dropdown import build_qti_with_questions
from pathlib import Path
from config import PART_USED

EVAL_TEMPLATE_PART = {
    1:"""
    You are an IELTS Reading Passage Rewriter.  
    Your task is to take an existing passage and rewrite it it so it better resembles an authentic IELTS Reading passage.

    ### Instructions:
    - Keep the **academic but accessible style** used in IELTS.
    - Revise according to the provided feedback categories.
    - Ensure the passage is about **850–950 words**, divided into **8 labeled paragraphs (A–H)**.
    - Maintain a **balance of sentence types**:
      - About 60% complex sentences
      - About 40% simple/compound sentences
      - Max sentence length: 35 words
      - Most sentence should be around 15 - 25 words, or less. Add more simple sentence so that the passage can be familiar with 6.0 - 7.0 students.
    - Vocabulary:
      - Target CEFR B2–C1
      - Academic Word List (AWL) should appear frequently
      - Limit rare/technical words to <5%, and define them if used
    - Readability:
      - Flesch Reading Ease: 40–60
      - Flesch–Kincaid Grade: 9–11
      - Stay strict along this range
    - Coherence: Logical flow, clear connections between ideas
    - Content:
      - Mix of factual explanation, examples, statistics, and one case study
      - Neutral, informative tone (not persuasive or emotional)

    ### Input:
    Original Passage:
    {passage}

    Feedbacks:
    {feedbacks}

    ### Output:
    The passage should be in the following format(the double quotes musn't be added):
    
"Text title: ...\n"
"Text: A. <text with advanced vocabulary and complex structures (>=90 words)>\n"
"Text: B. <text with advanced vocabulary and complex structures (>=90 words)>\n"
"Text: C. <text with advanced vocabulary and complex structures (>=90 words)>\n"
"Text: D. <text with advanced vocabulary and complex structures (>=90 words)>\n"
"Text: E. <text with advanced vocabulary and complex structures (>=90 words)>\n"
"Text: F. <text with advanced vocabulary and complex structures (>=90 words)>\n"
"Text: G. <text with advanced vocabulary and complex structures (>=90 words)>\n"
"Text: H. <text with advanced vocabulary and complex structures (>=90 words)>\n"
...
    ***Follow strictly the output format***
    """,
    2: """
    You are an IELTS Reading Passage Rewriter.  
    Your task is to take an existing passage and rewrite it so it better resembles an authentic IELTS Reading Part 2 passage.

    ### Instructions:
    - Style: Clear, descriptive, semi-academic (less technical than Part 1).
    - Length: **850 - 1000 words**, divided into **7 - 9 labeled paragraphs (A, B, C, ...)**.
    - Sentence balance:
      - About 50% complex sentences
      - About 50% simple/compound sentences
      - Max sentence length: 30 words
      - Most sentences should be 12–22 words, with some shorter for accessibility.
    - Vocabulary:
      - Target CEFR B1–B2 (upper-intermediate to low-advanced)
      - Frequent Academic Word List (AWL) usage
      - Limit rare/technical words to <3%, and explain them if included
    - Readability:
      - Flesch Reading Ease: 50–65
      - Flesch–Kincaid Grade: 8–10
    - Coherence: Strong topic development with smooth transitions
    - Content:
      - Mix factual explanation with descriptions, comparisons, and everyday examples
      - Include one short real-world case study or example
      - Maintain a **neutral, informative tone** (not persuasive or opinionated)

    ### Input:
    Original Passage:
    {passage}

    Feedbacks:
    {feedbacks}

    ### Output:
    The passage should be in the following format(the double quotes musn't be added):
    
Text title: ...
Text: A. <text with advanced vocabulary and complex structures (>=90 words)>
Text: B. <text with advanced vocabulary and complex structures (>=90 words)>
Text: C. <text with advanced vocabulary and complex structures (>=90 words)>
Text: D. <text with advanced vocabulary and complex structures (>=90 words)>
Text: E. <text with advanced vocabulary and complex structures (>=90 words)>
Text: F. <text with advanced vocabulary and complex structures (>=90 words)>
...

    ***Follow strictly the output format***
"""
,
    4: """
    Your task is to take an existing passage and rewrite it so it better resembles simplified IELTS-style reading passage for secondary students.
    📘 Passage Details (for simplified IELTS-style, secondary level)
🔹 Length & Structure

Word count: 350–400 words.

Paragraphs: 4 - 5 paragraphs.

Paragraph length: 3–5 sentences each.

Sentence length: Mostly 15–20 words, occasional shorter ones for clarity.

🔹 Paragraph Functions

1. Introduction:
- Present the global/cultural phenomenon.
- Explain why it is interesting or increasingly important worldwide.
- Use simple background (not heavy statistics).

2. Development / Institutions
- Show how governments, companies, or communities support it.
- Example of planning, training, or organisation.

3. Impact / Benefits
- Economic, social, or cultural benefits.
- Give clear examples (e.g., tourism growth, new jobs, popularity with young people).
4. Challenges / Criticisms
- At least two problems or criticisms.
- Use balanced phrasing: “Some people worry… Others argue…”.

5. Conclusion
- Restate influence in a balanced way.
- Suggest it will continue to grow or stay important in the future.

🔹 Vocabulary & Readability

Level: CEFR B1–B2 (secondary students).

Academic but simple:

Use words like global, cultural, economic, support, influence, challenge, benefit, critics, supporters.

Avoid rare or technical words (hegemony, commodification, paradigm).

Connectors: use common ones like however, therefore, in addition, for example, on the other hand.

Tone: Neutral, informative, slightly formal.

🔹 Language & Readability Rules

Use familiar academic connectors: “however”, “on the other hand”, “for example”, “therefore”.

Avoid idioms, proverbs, or overly metaphorical language.

Keep ideas concrete (examples from daily life, school, travel, youth culture).

Use hedging language but simple: “may”, “might”, “could”, “some people believe”.

🔹 Example Passage Opening (model sentence style)

"In recent years, street food has become a global trend. Once linked only with local markets, it is now celebrated in international festivals and travel shows. Many people see it not just as a meal, but as a way to experience culture directly."

Notice: short sentences, simple but not childish, clear subject-verb, easy vocabulary.


    ### Input:
    Original Passage:
    {passage}

    Feedbacks:
    {feedbacks}

    ### Output:
    The passage should be in the following format(the double quotes musn't be added):
    
Text title: ...
Text: A. <text>
Text: B. <text>
...

    ***Follow strictly the output format***
Follow all passage, vocabulary rules above.
"""
}
def rewrite_based_on(passage:str, feedbacks : str):
    

    client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE_URL)
    # attempts + validation loop
    max_attempts = 3
    threshold = 0.75  # passage validator score threshold
    last_traces = None

    final_passage = ""
    for attempt in range(1, max_attempts + 1):
        print("PASSAGE_REGENERATE attempt %d/%d", attempt, max_attempts)

        prompt = EVAL_TEMPLATE_PART[PART_USED].format(
    passage=passage,
    feedbacks=feedbacks
)
        response = client.chat.completions.create(
            model="gpt-5",  
            messages=[{"role": "developer", "content": "You are an IELTS Reading examiner assistant."}, {"role": "user", "content": prompt}] if PART_USED < 4 
                else [{"role": "developer", "content": "You are a Simplified IELTS Reading Passage examiner assistant (Secondary Student level Passage)."},
                        {"role": "user", "content": prompt}],
            temperature=0.7
        )
        # Extract the generated passage text
        final_passage = response.choices[0].message.content.strip()
        print(final_passage)
        # validate
        score, raw_traces, fb_traces = validate_passage_text(final_passage)
        print("Passage score=%.3f raw=%s feedback=%s", score, raw_traces, fb_traces)
        last_traces = raw_traces

        if score >= threshold:
            return final_passage
    
            # small wait/backoff to avoid rate-limit spikes
        time.sleep(1)
    print(f"PASSAGE generation failed to meet threshold after {max_attempts} attempts. Returning last result.")
    return final_passage
def generate_passage_with_rescoring(
    executors: Dict[str, Callable[[str, str, dict], Any]],
    base_prompts: Dict[str, str],
    topic: str,
    threshold: float = 0.8,
    max_attempts: int = 5):
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
        time.sleep(2)
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
        thresholds = {
            "passage": 0.8,
            "Vocabulary_Level": 0.85,
            "Sentence_Length_&_Grammar_Complexity": 0.85,
            "Authenticity_of_Style": 0.85,
            "Content_Balance": 0.85,
            "Readability": 0.85
        }

        if all(scores[k] >= thresholds[k] for k in thresholds):
            gen_questions(topic, outputs)
            return

        # Otherwise: build feedback and retry
        feedback_text = "; ".join(fb_traces) if fb_traces else "improve clarity and alignment with IELTS style"
        fb_prompt_suffix = f"\n\nFEEDBACK from evaluator: {feedback_text}"

        # Inject feedback into outputs for next executor call
        outputs["feedback"] = fb_prompt_suffix
        print(feedback_text)
        time.sleep(1)

    print(f"[RESCORE] Failed to reach threshold after {max_attempts} attempts, returning last result.")
    gen_questions(topic, outputs)


def save_generation_passages(topic: str, outputs: dict):
    # Sanitize topic name for filename
    safe_topic = topic.replace(" ", "_")

    # Count existing generations under "tests/"
    base_dir = f"tests/Part_{PART_USED}"
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
    lines.append(f"{passage}\n")
    textentry_questions = []
    matching_questions = []
    headings_questions = []
    last_type = ""
    # --- Questions ---
    if questions:
        if PART_USED in [1,2,3]:
            lines.append("Text: ANSWER THE FOLLOWING QUESTIONS\n")
        for i, q in enumerate(questions, 1):
            qtype = q.get("question_type", "").strip()
            if qtype == "True/False/Not Given" or qtype == "Yes/No/Not Given" or qtype == "MCQ":
                if last_type != qtype:
                    if qtype == "True/False/Not Given":
                        lines.append("Text: Do the following statements agree with the information given in the text? Write TRUE, FALSE, or NOT GIVEN.")
                    elif qtype == "Yes/No/Not Given":
                        lines.append("Text: Do the following statements agree with the information given in the text? Write YES, NO, or NOT GIVEN.")
                    elif qtype == "MCQ":
                        if PART_USED in [1,2,3]:
                            lines.append("Text: Choose the correct letter, A, B, C or D.")
                        else:
                            lines.append("Text: ANSWER THE FOLLOWING QUESTIONS. Choose the correct letter, A, B, C or D for each question.")
                    lines.append("")  # blank line
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
                headings_questions.append(q)
            elif qtype == "Sentence Completion":
                textentry_questions.append(q)
                continue
                # Short Answer (Answer the question in X words)
            elif qtype == "Summary Completion":
                textentry_questions.append(q)
                continue    
            elif qtype == "Short Answer":
                qtext = q.get("question_text", "").strip()
                answer = q.get("answer", "").strip()
                rationale = q.get("rationale", "").strip()

                lines.append(f"{i}. {qtext}")
                if rationale:
                    lines.append(f"... {rationale}")
                lines.append(f"* {answer}")
            elif qtype == "Matching Information":
                matching_questions.append(q)
            last_type = qtype
            
    # Write out
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    from tools import run_text2qti_and_extract
    run_text2qti_and_extract(Path(filename))

    output_zip = str(Path(filename).parent / "final_qti_with_questions.zip")
    from data_utils.dropdown import build_qti_with_questions
    build_qti_with_questions(txt_path=Path(filename), headings_questions=headings_questions, matching_questions=matching_questions, textentry_questions=textentry_questions, output_zip=output_zip)
    # insert them at the endx   
    print(f"[SAVED] {filename}")

def gen_questions(topic: str, outputs: Dict[str, Any]):
    out = questions_executor(prompt_template="", topic=topic, outputs_so_far=outputs)
    outputs["questions"] = out
    save_generation_passages(topic, outputs=outputs)