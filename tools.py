import subprocess, shutil
import sys
import zipfile
import os
from pathlib import Path

def run_text2qti_and_extract(txt_file: str):
    txt_folder = txt_file.parent

    # 1) Run text2qti
    subprocess.run([
        r"C:\Users\Dell\AppData\Local\Programs\Python\Python310\Scripts\text2qti.exe",
        txt_file
    ], check=True)



    # 2) Find the produced .zip file
    # text2qti names it like: <basename>_qti.zip
    zip_file = txt_folder / f"{txt_file.stem}.zip"

    if not zip_file.exists():
        print(f"Error: {zip_file} not found.")
        sys.exit(1)
    # 3) Create a temporary zip with fixed filename
    fixed_zip = txt_folder / f"{txt_file.stem}_qti_fixed.zip"
    with zipfile.ZipFile(zip_file, "r") as zin, zipfile.ZipFile(fixed_zip, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            path = Path(item.filename)

            # Keep folder structure the same
            if path.name.startswith("text2qti_assessment_") and path.suffix == ".xml":
                # rename only the big XML file
                new_path = path.parent / "text2qti_assessment.xml"
                zout.writestr(str(new_path), data)
            else:
                # copy everything else unchanged
                zout.writestr(item, data)

    # Replace original zip with fixed one
    zip_file.unlink()
    fixed_zip.rename(zip_file)

    # 4) Extract to target folder
    extract_dir = txt_folder / txt_file.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"✅ Extracted with renamed XML to {extract_dir}")