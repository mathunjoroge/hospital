import json
from pathlib import Path

files = ["clinical_relevance.json", "diagnosis_treatments.json"]
kb_dir = "/home/mathu/projects/hospital/departments/nlp/knowledge_base"

for file in files:
    file_path = Path(kb_dir) / file
    with open(file_path) as f:
        data = json.load(f)
    print(f"{file}: {data}")