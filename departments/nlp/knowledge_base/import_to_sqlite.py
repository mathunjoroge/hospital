import os
import json
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["clinical_knowledge_base"]

# JSON files to import
json_files = [
    "medical_stop_words.json",
    "medical_terms.json",
    "synonyms.json",
    "clinical_pathways.json",
    "history_diagnoses.json",
    "diagnosis_relevance.json",
    "management_config.json",
    "diagnosis_treatments.json"
]

# Set the folder where JSON files are located
base_path = "/home/mathu/projects/hospital/departments/nlp/knowledge_base"

for filename in json_files:
    filepath = os.path.join(base_path, filename)
    collection_name = filename.replace(".json", "")

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

            # Insert data based on structure
            if isinstance(data, list):
                db[collection_name].insert_many(data)
            elif isinstance(data, dict):
                # Store as individual document
                db[collection_name].insert_one(data)
            else:
                print(f"Unknown format in {filename}")
            print(f"✅ Imported {filename} into collection '{collection_name}'")

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")

print("🚀 All done.")
