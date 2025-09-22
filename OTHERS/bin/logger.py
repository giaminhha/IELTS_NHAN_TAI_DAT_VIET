# logger.py
import json
from pathlib import Path
from datetime import datetime

class GEPA_Logger:
    def __init__(self, log_dir="logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(log_dir) / f"gepa_run_{timestamp}.jsonl"

    def log(self, record: dict):
        """Append one record as JSON line"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
