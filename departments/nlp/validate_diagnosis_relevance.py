import json
from pathlib import Path

file_path = "/home/mathu/projects/hospital/departments/nlp/knowledge_base/diagnosis_relevance.json"
try:
    with open(file_path) as f:
        data = json.load(f)
    print(f"Content: {data}")
    if isinstance(data, list):
        valid_items = [
            item for item in data
            if isinstance(item, dict) and "diagnosis" in item and "relevance" in item
        ]
        print(f"Valid items: {len(valid_items)}")
        print(f"Sample valid items: {valid_items[:2]}")
        invalid_items = [
            item for item in data
            if not (isinstance(item, dict) and "diagnosis" in item and "relevance" in item)
        ]
        print(f"Invalid items: {len(invalid_items)}")
        print(f"Sample invalid items: {invalid_items[:2]}")
    else:
        print("Data is not a list")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")