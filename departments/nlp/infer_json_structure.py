import os
import json
from collections import defaultdict

def infer_document_structure(doc, structure=None, path=''):
    if structure is None:
        structure = {}

    for key, value in doc.items():
        full_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            structure[full_path] = 'object'
            infer_document_structure(value, structure, full_path)
        elif isinstance(value, list):
            structure[full_path] = 'array'
            if value and isinstance(value[0], dict):
                infer_document_structure(value[0], structure, full_path + '[]')
        else:
            structure[full_path] = type(value).__name__

    return structure

def infer_schema_from_json_file(filepath, sample_size=10):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse {filepath}: {e}")
            return {}

    data = data[:sample_size]
    combined_structure = defaultdict(set)

    for doc in data:
        if isinstance(doc, dict):
            structure = infer_document_structure(doc)
            for field, dtype in structure.items():
                combined_structure[field].add(dtype)

    schema = {field: list(dtypes) for field, dtypes in combined_structure.items()}
    return schema

def main():
    base_dir = "/home/mathu/projects/hospital/departments/nlp/knowledge_base"

    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(base_dir, filename)
            print(f"\n📄 File: {filename}")
            schema = infer_schema_from_json_file(filepath)
            if schema:
                print(json.dumps(schema, indent=2))
            else:
                print("⚠️  No valid documents found or failed to infer schema.")

if __name__ == "__main__":
    main()
