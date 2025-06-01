import json
from pathlib import Path

input_file = "/home/mathu/projects/hospital/departments/nlp/knowledge_base/diagnosis_relevance.json"
output_file = "/home/mathu/projects/hospital/departments/nlp/knowledge_base/diagnosis_relevance_transformed.json"

# Simple category mapping (customize as needed)
category_mapping = {
    "weight loss": "general",
    "inattention": "neurological",
    "hemorrhage": "hematologic",
    "delirium": "neurological",
    "pruritus": "dermatologic",
    "meniscal tear": "musculoskeletal",
    "cervical strain": "musculoskeletal",
    "angina": "cardiovascular",
    "arrhythmia": "cardiovascular",
    "aortic dissection": "cardiovascular",
    "pericarditis": "cardiovascular",
    "chronic kidney disease": "renal",
    "acute kidney injury": "renal",
    "bacterial vaginosis": "gynecologic",
    "candidiasis": "gynecologic",
    "chlamydia": "gynecologic",
    "gonorrhea": "gynecologic",
    "trigeminal neuralgia": "neurological",
    "cluster headache": "neurological",
    "temporal arteritis": "rheumatologic",
    "malaria": "infectious",
    "leptospirosis": "infectious",
    "mononucleosis": "infectious",
    "bronchiolitis": "respiratory",
    "croup": "respiratory",
    "barrett's esophagus": "gastrointestinal",
    "diverticulosis": "gastrointestinal"
}

try:
    with open(input_file) as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        print("Data is not a dictionary")
        exit(1)
    
    transformed = [
        {
            "diagnosis": diagnosis,
            "relevance": symptoms,
            "category": category_mapping.get(diagnosis, "unknown")
        }
        for diagnosis, symptoms in data.items()
        if isinstance(symptoms, list)
    ]
    
    with open(output_file, "w") as f:
        json.dump(transformed, f, indent=2)
    
    print(f"Transformed data written to {output_file}")
    print(f"Records: {len(transformed)}")
    print(f"Sample: {transformed[:2]}")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
except Exception as e:
    print(f"Error: {e}")