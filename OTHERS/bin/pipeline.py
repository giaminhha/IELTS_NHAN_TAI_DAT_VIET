# pipeline.py
from executors import (
    generate_passage,
    generate_questions,
    generate_distractors,
    check_penmanship,
)


def run_pipeline():
    topic = "Climate Change and Polar Ice Melt"
    sources = [
        "[S1] Recent studies on polar ice sheet reduction.",
        "[S2] Satellite data analysis of Arctic sea ice.",
        "[S3] Impact of melting glaciers on global sea levels.",
    ]

    # ---- Step 1: Passage ----
    print("=== Generating Passage ===")
    passage = generate_passage(topic, sources)
    print(passage)
    print("\n")

    # ---- Step 2: Questions ----
    qtype_id = "Multiple_Choice"  # you can try others too
    print(f"=== Generating {qtype_id} Questions ===")
    questions = generate_questions(passage, qtype_id)
    for q in questions:
        print(q)
    print("\n")

    # ---- Step 3: Distractors ----
    print("=== Generating Distractors ===")
    distractors = generate_distractors(passage)
    for d in distractors:
        print(d)
    print("\n")

    # ---- Step 4: Penmanship Check (example) ----
    print("=== Checking Penmanship ===")
    answer = "The polar ice caps are shrinking due to rising global temperatures."
    result = check_penmanship(answer)
    print(result)


if __name__ == "__main__":
    run_pipeline()
