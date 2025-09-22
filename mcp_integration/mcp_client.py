# mcp_client.py
import requests
from typing import Any, Dict, List

class MCPClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> Any:
        resp = requests.get(f"{self.base_url}{path}")
        resp.raise_for_status()
        return resp.json()

    def get_passage_rules(self) -> List[Dict]:
        return self._get("/get_passage_rules")

    def get_question_type_context(self, qtype_id: str) -> Dict:
        return self._get(f"/get_question_type_context/{qtype_id}")

    def get_distractor_patterns(self) -> List[Dict]:
        return self._get("/get_distractor_patterns")

    def get_penmanship_rules(self) -> List[Dict]:
        return self._get("/get_penmanship_rules")
    def get_passage_examples(self) -> List[Dict]:
        return self._get("/get_passage_examples")

    def get_question_examples(self, qtype_id: str = None) -> List[Dict]:
        if qtype_id:
            return self._get(f"/get_question_examples/{qtype_id}")
        return self._get("/get_question_examples")
