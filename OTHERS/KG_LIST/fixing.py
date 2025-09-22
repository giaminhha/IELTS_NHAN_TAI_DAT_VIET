import json

with open("lists.json", "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]

# Category → type mapping
categories = {
    "format_rules": "FormatRule",
    "writing_styles": "WritingStyle",
    "skills": "Skill",
    "distractors": "Distractor",
    "penmanship": "Penmanship",
    "question_types": "QuestionType",
    "question_type_rules": "QuestionTypeRule",
    "answer_behaviours": "AnswerBehaviour",
    "example_patterns": "ExamplePattern"
}

for n in nodes:
    # Top node already has type
    for cat, label in categories.items():
        if cat in n:
            for child in n[cat]:
                child["type"] = label
                # Add subrules if any
                if "subrules" in child:
                    for sub in child["subrules"]:
                        sub["type"] = label + "Subrule"

with open("lists_with_types.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
