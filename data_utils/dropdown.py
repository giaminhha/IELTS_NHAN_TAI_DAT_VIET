#!/usr/bin/env python3
"""
dropdown.py

Workflow (Modified):
 - Locates an existing folder named after the input .txt file.
 - Inserts questions (textentry or dropdown) into the text2qti_assessment_*.xml inside that folder.
 - Writes rationale into qtimetadata and itemfeedback.
 - Places items in numeric order by QID.
 - Rezips the folder -> final_zip (preserving the top-level folder name).
"""

import os
import re
import zipfile
import shutil
import subprocess
import xml.etree.ElementTree as ET
import random
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
from canvasapi import Canvas
import json
import sys, os

# Add parent folder (tests/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import API_URL, API_KEY, COURSE_ID, PART_USED


# ... (All other utility functions like sanitize_name, run_text2qti, etc. are unchanged) ...
def sanitize_name(name: str) -> str:
    name = name.strip()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)

def run_text2qti(txt_path: str) -> str:
    parent = os.path.dirname(txt_path) or "."
    basename = os.path.basename(txt_path)
    try:
        subprocess.run(["text2qti", basename], cwd=parent, check=True)
    except FileNotFoundError:
        raise RuntimeError("text2qti not found. Install it or add to PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"text2qti failed: {e}")
    zips = []
    for f in os.listdir(parent):
        if f.endswith(".zip") and "text2qti_assessment" in f:
            zips.append(os.path.join(parent, f))
    if not zips:
        base_no_ext = os.path.splitext(basename)[0]
        for f in os.listdir(parent):
            if f.endswith(".zip") and base_no_ext in f:
                zips.append(os.path.join(parent, f))
    if not zips:
        raise FileNotFoundError("No QTI zip produced by text2qti found in the txt file folder.")
    zips.sort(key=lambda p: os.path.getmtime(p))
    return zips[-1]

def safe_extract_zip(zip_path: str, temp_root: str, verbose: bool = True):
    if os.path.exists(temp_root): shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            target_path = os.path.join(temp_root, member.filename)
            if not os.path.abspath(target_path).startswith(os.path.abspath(temp_root)):
                continue
            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                if os.path.isdir(target_path): continue
                parent_dir = os.path.dirname(target_path)
                if not os.path.isdir(parent_dir): os.makedirs(parent_dir, exist_ok=True)
                with z.open(member, 'r') as source_file, open(target_path, 'wb') as target_file:
                    shutil.copyfileobj(source_file, target_file)
    return temp_root

# --- [CORRECTED FUNCTION] ---
# The logic here is updated to be more robust against filenames with hidden whitespace.
def find_assessment_xml_in_base(base_folder: str) -> str:
    for root, _, files in os.walk(base_folder):
        for f in files:
            # Clean the filename by removing leading/trailing whitespace
            clean_f = f.strip()
            if clean_f.startswith("text2qti_assessment") and clean_f.endswith(".xml") and "meta" not in clean_f:
                # Return the path constructed with the CLEAN filename
                return os.path.join(root, clean_f)
    raise FileNotFoundError(f"Main QTI XML (text2qti_assessment*.xml) not found in folder: {base_folder}")

def rezip_folder_contents(base_folder_path: str, output_zip: str) -> str:
    """
    Zip only the *contents* of base_folder_path (files and subfolders),
    and include empty directories (so non_cc_assessments/ is preserved).
    """
    base_folder_path = os.path.abspath(base_folder_path)
    if not os.path.isdir(base_folder_path):
        raise FileNotFoundError(f"Base folder does not exist: {base_folder_path}")
    if os.path.exists(output_zip):
        os.remove(output_zip)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(base_folder_path):
            # add a directory entry for this folder (skip the top-level base folder itself)
            rel_root = os.path.relpath(root, base_folder_path)
            if rel_root != ".":
                # ensure forward slashes inside zip
                dir_entry = rel_root.replace(os.sep, "/").rstrip("/") + "/"
                # write an empty directory entry (this ensures empty dirs are present)
                z.writestr(dir_entry, "")

            # add files (normal behavior)
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, base_folder_path).replace(os.sep, "/")
                z.write(abs_path, rel_path)

    return output_zip




def get_namespace(root: ET.Element) -> Optional[str]:
    m = re.match(r"\{(.+)\}", root.tag)
    return m.group(1) if m else None

# -----------------------
# Builders and other helpers (Unchanged)
# -----------------------
def build_textentry_item(question: Dict, ns_uri: Optional[str]) -> ET.Element:
    def tag(t):
        return f"{{{ns_uri}}}{t}" if ns_uri else t

    answers = question.get("answer", {}) or {}
    num_blanks = len(answers)

    # Create item element and set points_possible attribute to number of blanks
    item_attrs = {"ident": question["id"], "title": "Question", "points_possible": f"{num_blanks}.0"}
    item = ET.Element(tag("item"), item_attrs)

    # ----------------- Metadata -----------------
    itemmetadata = ET.SubElement(item, tag("itemmetadata"))
    qtimetadata = ET.SubElement(itemmetadata, tag("qtimetadata"))

    def add_meta(label, entry):
        field = ET.SubElement(qtimetadata, tag("qtimetadatafield"))
        ET.SubElement(field, "fieldlabel").text = label
        ET.SubElement(field, "fieldentry").text = entry

    add_meta("question_type", "fill_in_multiple_blanks_question")
    # points_possible now reflects the number of blanks (as you requested)
    add_meta("points_possible", f"{num_blanks}.0")

    # ----------------- Presentation -----------------
    presentation = ET.SubElement(item, tag("presentation"))
    material = ET.SubElement(presentation, tag("material"))
    q_text = question.get("question_text", "")
    # replace occurrences like "9____" with "[blank9]" (keeps inline placeholder)

    # --- split into first sentence vs remaining ---
    # Regex: split once at the first period followed by space
    parts = re.split(r'(?<=\.)\s+', q_text, maxsplit=1)

    if len(parts) == 2:
        first_sentence, rest = parts
        formatted_text = f"<div><p>{first_sentence}</p><p>{rest}</p></div>"
    else:
        # if only one sentence, just wrap once
        formatted_text = f"<div><p>{q_text}</p></div>"

    for blank_id in answers.keys():
        q_text = q_text.replace(f"{blank_id}____", f"[blank{blank_id}]")
    mattext = ET.SubElement(material, tag("mattext"), {"texttype": "text/html"})
    mattext.text = f"<div><div>{formatted_text}</div></div>"

    # create response choices (one correct label per blank)
    correct_idents = {}
    for blank_id, correct_answer in answers.items():
        response_lid = ET.SubElement(presentation, tag("response_lid"), {"ident": f"response_blank{blank_id}"})
        # material inside response_lid (label text)
        ET.SubElement(ET.SubElement(response_lid, tag("material")), tag("mattext")).text = f"blank{blank_id}"
        render_choice = ET.SubElement(response_lid, tag("render_choice"))

        # generate an ident (you said IDs don't matter)
        primary_ident = str(random.randint(1000, 9999))
        correct_idents[blank_id] = primary_ident

        response_label = ET.SubElement(render_choice, tag("response_label"), {"ident": primary_ident})
        mat = ET.SubElement(response_label, tag("material"))
        ET.SubElement(mat, tag("mattext"), {"texttype": "text/plain"}).text = correct_answer

    # ----------------- Resprocessing -----------------
    resprocessing = ET.SubElement(item, tag("resprocessing"))
    outcomes = ET.SubElement(resprocessing, tag("outcomes"))
    # decvar maxvalue set to number of blanks so SCORE is on the same scale
    ET.SubElement(outcomes, tag("decvar"), {
        "maxvalue": f"{num_blanks}.0",
        "minvalue": "0",
        "varname": "SCORE",
        "vartype": "Decimal"
    })

    # each blank yields 1 point (so total = num_blanks)
    if num_blanks > 0:
        for blank_id, correct_ident in correct_idents.items():
            respcondition = ET.SubElement(resprocessing, tag("respcondition"))
            conditionvar = ET.SubElement(respcondition, tag("conditionvar"))
            varequal = ET.SubElement(conditionvar, tag("varequal"), {"respident": f"response_blank{blank_id}"})
            varequal.text = correct_ident
            setvar = ET.SubElement(respcondition, tag("setvar"), {"varname": "SCORE", "action": "Add"})
            setvar.text = f"{1:.2f}"
    else:
        # no blanks -> no respconditions; decvar maxvalue is 0.0
        pass

    # ----------------- Feedback (general_fb) -----------------
    rationales = question.get("rationale") or {}
    if rationales:
        feedback = ET.SubElement(item, tag("itemfeedback"), {"ident": "general_fb"})
        flow_mat = ET.SubElement(feedback, tag("flow_mat"))
        material = ET.SubElement(flow_mat, tag("material"))
        mattext_fb = ET.SubElement(material, tag("mattext"), {"texttype": "text/html"})
        fb_text = "<div>" + " ".join(f"<div><span>{txt}</span></div>" for txt in rationales.values()) + "</div>"
        mattext_fb.text = fb_text

    return item

import xml.etree.ElementTree as ET
import random
from typing import Dict, Optional

def build_matching_item(question: Dict, ns_uri: Optional[str] = None) -> ET.Element:
    """
    Build a QTI 'matching' item XML from a question dict, including rationale.
    question = {
        "id": "Q1",
        "question_text": "Which paragraph contains the following information? Write the correct letter.",
        "questions": {
            "1": "An example of a British park that influenced others.",
            "2": "A concern that green spaces might lead to unrest.",
            "3": "The idea that exposure to plants and animals could teach moral lessons."
        },
        "options": ["A", "B", "C", "D", "E", "F", "G"],
        "answers": { "1": "B", "2": "F", "3": "D" },
        "rationale": {     # optional
            "1": "Paragraph B mentions the park example.",
            "2": "Paragraph F discusses unrest.",
            "3": "Paragraph D talks about moral lessons."
        }
    }
    """
    def tag(t):
        return f"{{{ns_uri}}}{t}" if ns_uri else t

    sub_questions = question["questions"]
    options = question["options"]
    answers = question["answers"]
    total_points = len(sub_questions)

    # --- Item root ---
    item_attrs = {"ident": question["id"], "title": "Question", "points_possible": f"{total_points}.0"}
    item = ET.Element(tag("item"), item_attrs)

    # --- Metadata ---
    itemmetadata = ET.SubElement(item, tag("itemmetadata"))
    qtimetadata = ET.SubElement(itemmetadata, tag("qtimetadata"))
    def add_meta(label, entry):
        field = ET.SubElement(qtimetadata, tag("qtimetadatafield"))
        ET.SubElement(field, tag("fieldlabel")).text = label
        ET.SubElement(field, tag("fieldentry")).text = entry

    add_meta("question_type", "matching_question")
    add_meta("points_possible", f"{total_points}.0")
    add_meta("original_answer_ids", ",".join([str(random.randint(1000, 9999)) for _ in sub_questions]))

    # --- Presentation ---
    presentation = ET.SubElement(item, tag("presentation"))
    material = ET.SubElement(presentation, tag("material"))
    ET.SubElement(material, tag("mattext"), {"texttype": "text/html"}).text = question["question_text"]

    ident_map = {}
    for qid, qtext in sub_questions.items():
        resp_lid = ET.SubElement(presentation, tag("response_lid"), {"ident": f"response_{qid}"})
        mat = ET.SubElement(resp_lid, tag("material"))
        ET.SubElement(mat, tag("mattext"), {"texttype": "text/plain"}).text = qtext

        render_choice = ET.SubElement(resp_lid, tag("render_choice"))
        for opt in options:
            resp_label_ident = str(random.randint(1000, 9999))
            resp_label = ET.SubElement(render_choice, tag("response_label"), {"ident": resp_label_ident})
            mat_opt = ET.SubElement(resp_label, tag("material"))
            ET.SubElement(mat_opt, tag("mattext"), {"texttype": "text/plain"}).text = opt
            if opt == answers[qid]:
                ident_map[qid] = resp_label_ident

    # --- Resprocessing ---
    resprocessing = ET.SubElement(item, tag("resprocessing"))
    outcomes = ET.SubElement(resprocessing, tag("outcomes"))
    ET.SubElement(outcomes, tag("decvar"), {
        "maxvalue": f"{total_points}",
        "minvalue": "0",
        "varname": "SCORE",
        "vartype": "Decimal"
    })

    for qid, correct_ident in ident_map.items():
        respcond = ET.SubElement(resprocessing, tag("respcondition"))
        condvar = ET.SubElement(respcond, tag("conditionvar"))
        varequal = ET.SubElement(condvar, tag("varequal"), {"respident": f"response_{qid}"})
        varequal.text = correct_ident
        setvar = ET.SubElement(respcond, tag("setvar"), {"varname": "SCORE", "action": "Add"})
        setvar.text = "1.00"

    # --- Feedback / Rationale ---
    rationales = question.get("rationale") or {}
    if rationales:
        feedback = ET.SubElement(item, tag("itemfeedback"), {"ident": "general_fb"})
        flow_mat = ET.SubElement(feedback, tag("flow_mat"))
        material = ET.SubElement(flow_mat, tag("material"))
        fb_text = "<div>" + " ".join(f"<div><span>{txt}</span></div>" for txt in rationales.values()) + "</div>"
        ET.SubElement(material, tag("mattext"), {"texttype": "text/html"}).text = fb_text

    return item


def insert_items_into_assessment(xml_path: Path, items: list[str], mode="textentry") -> None:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    ns = get_namespace(root)

    section = root.find(f".//{{{ns}}}section[@ident='root_section']") or root.find(f".//{{{ns}}}section")
    if section is None:
        raise ValueError("No <section> found in assessment XML")

    for item_str in items:
        item_elem = ET.fromstring(item_str)
        section.append(item_elem)

    # Ensure directory exists before writing
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    # Inside insert_items_into_assessment
    section_dir = xml_path.parent
    section_dir.mkdir(parents=True, exist_ok=True)  # <-- this ensures folder exists
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def upload_qti_to_canvas(file_path: str, reading_task_num: int):
    payload = {
        "migration_type": "qti_converter",
        "pre_attachment": {"name": "qti.zip"},
        "settings": {"import_quizzes_next": False}
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Step 1: Create migration
    response = requests.post(
        f"{API_URL}/api/v1/courses/{COURSE_ID}/content_migrations/",
        headers=headers,
        data=json.dumps(payload)
    )
    response.raise_for_status()
    migration = response.json()
    upload_url = migration["pre_attachment"]["upload_url"]
    upload_params = migration["pre_attachment"]["upload_params"]

    # Step 2: Upload QTI zip
    with open(file_path, "rb") as file:
        files = {"file": (upload_params["Filename"], file)}
        upload_response = requests.post(upload_url, data=upload_params, files=files)
        upload_response.raise_for_status()

    # Step 3: Poll migration progress
    progress_url = migration["progress_url"]
    while True:
        progress_response = requests.get(progress_url, headers=headers)
        progress_response.raise_for_status()
        progress = progress_response.json()
        print(f"\rMigration progress: {progress['completion']}%", end="\n")
        if progress["workflow_state"] == "completed":
            print("\nMigration completed successfully.")
            break
        elif progress["workflow_state"] == "failed":
            print("\nMigration failed.")
            return None
        time.sleep(3)

    # Step 4: Find quiz by name
    canvas = Canvas(API_URL, API_KEY)
    course = canvas.get_course(COURSE_ID)
    quizzes = course.get_quizzes()

    # Base name (without number)
    base_name = f"[Reading] IELTS Reading Part {reading_task_num} Quiz"
    matching_quizzes = [q for q in quizzes if q.title.startswith(base_name)]

    if matching_quizzes:
        # If already exists → find the max quiz number
        numbers = []
        for q in matching_quizzes:
            parts = q.title.split()
            try:
                numbers.append(int(parts[-1]))  # last token is the number
            except ValueError:
                pass
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1

    # Step 5: Grab the newest quiz (the one we just imported)
    new_quiz = max(quizzes, key=lambda q: q.id, default=None)

    if new_quiz:
        new_title = f"{base_name} {next_num}"
        new_quiz.edit(quiz={'title': new_title})
        print(f"Renamed quiz: {new_title}")

        new_quiz.edit(quiz={'published': True})
        print("Quiz has been published.")

        return f"{API_URL}/courses/{COURSE_ID}/quizzes/{new_quiz.id}"
    else:
        print("No new quiz was created.")
        return None

def build_text_only_item(question: Dict, ns_uri: Optional[str] = None) -> ET.Element:
    """
    Build a text_only_question item that just displays text (e.g., headings list).
    question = {
        "id": "H1_headings",
        "first_displayed": [
            "LIST OF HEADINGS",
            "i. Heading 1",
            "ii. Heading 2",
            "iii. Heading 3"
        ]
    }
    """
    def tag(t): return f"{{{ns_uri}}}{t}" if ns_uri else t

    item_attrs = {"ident": question["id"], "title": "Question", "points_possible": "0"}
    item = ET.Element(tag("item"), item_attrs)

    # Metadata
    itemmetadata = ET.SubElement(item, tag("itemmetadata"))
    qtimetadata = ET.SubElement(itemmetadata, tag("qtimetadata"))
    for label, entry in [
        ("question_type", "text_only_question"),
        ("points_possible", "0"),
        ("original_answer_ids", ",,,")
    ]:
        field = ET.SubElement(qtimetadata, tag("qtimetadatafield"))
        ET.SubElement(field, tag("fieldlabel")).text = label
        ET.SubElement(field, tag("fieldentry")).text = entry

    # Presentation
    presentation = ET.SubElement(item, tag("presentation"))
    material = ET.SubElement(presentation, tag("material"))
    mattext = ET.SubElement(material, tag("mattext"), {"texttype": "text/html"})
    mattext.text = "<div>" + "".join(f"<p>{h}</p>" for h in question["first_displayed"]) + "</div>"

    return item


# -----------------------
# top-level pipeline (Unchanged)
# -----------------------
def build_qti_with_questions(txt_path: str,
                             textentry_questions: List[Dict],
                             matching_questions: List[Dict],
                             headings_questions: List[Dict],
                             output_zip: str = "final_qti.zip") -> str:
    txt_path = Path(txt_path)
    parent = txt_path.parent
    target_base = parent

    print(f"1) Locating target content folder: {target_base}")
    if not target_base.is_dir():
        raise FileNotFoundError(f"The required folder '{target_base}' was not found.")
    print("   ✅ Folder found.")

    # --- locate assessment xml ---
    assessment_xml = Path(find_assessment_xml_in_base(str(target_base)))
    print("2) Found assessment xml:", assessment_xml)

    # --- Parse XML ---
    tree = ET.parse(str(assessment_xml))
    root = tree.getroot()
    ns_uri = get_namespace(root)
    section = root.find(f".//{{{ns_uri}}}section[@ident='root_section']") or root.find(f".//{{{ns_uri}}}section")
    if section is None:
        raise ValueError("No <section> found in assessment XML")

    # --- Build heading question items ---
    # --- Build heading question items (insert at start) ---
    if headings_questions:
        items_xml = []
        for hq in headings_questions:
            # 1) text_only_question showing headings
            text_only_q = {
                "id": f"{hq['id']}_headings",
                "first_displayed": [
                    "LIST OF HEADINGS",
                    hq["first_displayed"]
                ]
            }
            text_only_elem = build_text_only_item(text_only_q, ns_uri)
            items_xml.append(text_only_elem)

            # 2) actual matching question using headings as options
            matching_elem = build_matching_item(hq, ns_uri)
            items_xml.append(matching_elem)

        # insert all at the start in reverse order
        for item_elem in reversed(items_xml):
            section.insert(0, item_elem)


    # --- Build normal matching question items (insert at the start) ---
    if matching_questions:
        items_xml = []
        for q in matching_questions:
            item_elem = build_matching_item(q, ns_uri)
            items_xml.append(item_elem)

        if items_xml:
            for item_elem in reversed(items_xml):
                section.insert(0, item_elem)

    # --- Build textentry question items (append at the end) ---
    if textentry_questions:
        for q in textentry_questions:
            item_elem = build_textentry_item(q, ns_uri)
            section.append(item_elem)

    # --- Write back to XML ---
    xml_path = assessment_xml
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print("   ✅ XML modification complete.")

    # --- Re-zip folder contents ---
    print("3) Re-zipping target base folder to:", output_zip)
    final = rezip_folder_contents(str(txt_path.parent), output_zip)
    print("✅ Done. Final zip created:", final)

    # --- Upload to Canvas ---
    print("4) Uploading to Canvas...")
    upload_qti_to_canvas(final, reading_task_num=PART_USED)
    return final
