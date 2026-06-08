import json
from pathlib import Path


class MetadataStore:
    def __init__(self):
        self.records = []

    def add_records(self, chunk_records: list[dict]):
        self.records.extend(chunk_records)

    def get_record(self, idx: int) -> dict:
        return self.records[idx]

    def save(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    @staticmethod
    def load(input_path: str):
        path = Path(input_path)
        store = MetadataStore()
        store.records = json.loads(path.read_text(encoding="utf-8"))
        return store