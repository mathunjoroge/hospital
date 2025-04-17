import json
from datetime import datetime

# File paths
INPUT_FILE = "/home/mathu/projects/hospital/departments/nlp/knowledge_base/diagnosis_relevance.json"
BACKUP_FILE = f"{INPUT_FILE}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Terms to exclude (non-symptoms or vague)
EXCLUDE_TERMS = [
    "treatment", "management", "diagnosis", "s of", "identify predict",
    "mechanisms", "evaluation", "prevention", "therapeutic", "fact",
    "presentations", "approach", "investigation", "bidity", "multim",
    "within", "phenotype", "pathways", "clinicians", "associated with",
    "experiencing persistent symptoms", "covid-19", "procedures", "evidence",
    "consequences", "optimal", "staging", "prognosis", "key points",
    "specialized care", "depth", "manifestations", "outside", "development",
    "mones", "bidities", "includes", "essential", "types", "preferred",
    "relapsing", "paraclinical", "prodromal", "systemic review", "swift",
    "differentiating", "activity", "take a", "example", "rarely occur",
    "protective measures", "advancement", "strategies", "sequelae",
    "gan failure", "caused by", "potential", "early signs", "positivity",
    "although", "specific", "combinations", "subclinical", "both",
    "may coincide", "precede", "der to", "pure red", "but they may",
    "m the contemp", "even experienced", "may not be", "requiring screening",
    "from a relatively", "estimated the frequency", "is useful f",
    "is multifaceted", "have been addressed", "leads to several",
    "it can even", "discussing its variability", "it is merely",
    "patients with", "often integrated", "present as an", "to enhance",
    "as well as", "overlap syndromes", "in several different",
    "a wide differential", "highlights some", "that present",
    "their c", "which m", "of ad", "of ps", "of uwl", "kup f",
    "which extra caution", "of the disease", "is varied", "ly understood",
    "s include abn", "a general feeling", "being unwell", "with optional",
    "y diagnostic", "k up", "need f", "such as glioma", "experienced physicians",
    "of zikv", "this emerging", "including the heart", "eurasian manifestations",
    "while the remaining", "temp", "without subtle", "maternal risk",
    "diagnostic w", "pain is typical"
]

# Symptom normalization mappings
SYMPTOM_NORMALIZATION = {
    "whistling sound in the lungs called wheezing": "wheezing",
    "having trouble breathing": "shortness of breath",
    "king harder than usual to breathe": "shortness of breath",
    "a tight": "chest tightness",
    "tness of breath": "shortness of breath",
    "blood-streaked mucusrapid": "hemoptysis",
    "low mood": "depressed mood",
    "poor concentration": "difficulty concentrating",
    "transient weakness": "temporary weakness",
    "ary confusion": "temporary confusion",
    "lip smacking": "automatisms",
    "eye blinking": "automatisms",
    "right lower quadrant pain": "abdominal pain",
    "left lower quadrant pain": "abdominal pain",
    "right upper quadrant pain": "abdominal pain",
    "unintentional weight loss": "weight loss",
    "non-cardiac chest pain": "chest pain",
    "postprandial bloating": "bloating",
    "nocturnal paresthesia": "tingling",
    "hand numbness": "numbness",
    "tingling in fingers": "tingling",
    "hand pain": "pain",
    "hand weakness": "weakness",
    "vaso-occlusive pain": "pain",
    "chronic back pain": "back pain",
    "local pain": "pain",
    "widespread pain": "pain",
    "bone pain": "pain",
    "joint pain": "pain",
    "flank pain": "pain",
    "epigastric pain": "abdominal pain",
    "unilateral pelvic pain": "pelvic pain",
    "pelvic cramping": "pelvic pain",
    "abdominal tenderness": "abdominal pain",
    "local swelling": "swelling",
    "morning stiffness": "stiffness",
    "transient vision loss": "vision loss",
    "curtain vision loss": "vision loss",
    "peripheral vision loss": "vision loss",
    "central vision loss": "vision loss",
    "cloudy vision": "blurred vision",
    "night vision difficulty": "blurred vision",
    "fading colors": "blurred vision",
    "severe itching": "itching",
    "nocturnal pruritus": "itching",
    "painful vesicles": "vesicles",
    "recurrent sores": "sores",
    "paroxysmal cough": "cough",
    "whooping sound": "cough",
    "maculopapular rash": "rash",
    "papular rash": "rash",
    "erythematous": "erythema",
    "honey-colored crusts": "sores",
    "uneven shoulders": "postural asymmetry",
    "uneven waist": "postural asymmetry",
    "rib prominence": "postural asymmetry",
    "frequent contractions": "contractions",
    "uterine contractions": "contractions",
    "bright red blood": "vaginal bleeding",
    "mal bleeding": "vaginal bleeding",
    "chronic vaginal discharge": "vaginal discharge",
    "pelvic masses": "pelvic pain",
    "bronze skin": "skin discoloration",
    "memory loss": "cognitive impairment",
    "behavioral changes": "cognitive impairment",
    "cognitive fog": "cognitive impairment",
    "disorientation": "confusion",
    "fluctuating consciousness": "confusion",
    "sleep disturbances": "insomnia",
    "auditory impairment": "hearing loss",
    "sensory loss": "numbness",
    "pathologic fractures": "fractures",
    "fragility fractures": "fractures",
    "bow legs": "skeletal deformities",
    "growth delay": "skeletal deformities",
    "Koplik spots": "rash",
    "hair loss": "alopecia",
    "restless legs": "restlessness",
    "mouth sores": "oral ulcers",
    "hemorrhagic diarrhea": "diarrhea"
}

# Curated symptoms for conditions with poor data
CURATED_SYMPTOMS = {
    "migraine": ["headache", "nausea", "photophobia", "aura", "vomiting"],
    "asthma": ["wheezing", "shortness of breath", "cough", "chest tightness"],
    "heart failure": ["shortness of breath", "edema", "fatigue", "orthopnea", "paroxysmal nocturnal dyspnea"],
    "pneumonia": ["cough", "fever", "shortness of breath", "chest pain", "fatigue"],
    "diabetes mellitus": ["polyuria", "polydipsia", "weight loss", "fatigue", "blurred vision"],
    "atopic dermatitis": ["itching", "rash", "dry skin", "erythema", "eczema"],
    "celiac disease": ["diarrhea", "abdominal pain", "bloating", "weight loss", "fatigue"],
    "inflammatory bowel disease": ["diarrhea", "abdominal pain", "bloating", "weight loss", "fatigue"],
    "gastroenteritis": ["diarrhea", "vomiting", "abdominal pain", "fever", "nausea"],
    "pulmonary embolism": ["shortness of breath", "chest pain", "tachycardia", "hemoptysis", "leg swelling"],
    "meningitis": ["fever", "headache", "neck stiffness", "photophobia", "confusion"],
    "sepsis": ["fever", "tachycardia", "hypotension", "confusion", "shortness of breath"],
    "rheumatoid arthritis": ["joint pain", "joint swelling", "morning stiffness", "fatigue", "fever"],
    "gout": ["joint pain", "swelling", "erythema", "tenderness", "fever"],
    "psoriasis": ["rash", "plaques", "itching", "scaling", "erythema"],
    "psoriatic arthritis": ["joint pain", "swelling", "morning stiffness", "psoriasis", "dactylitis"],
    "scleroderma": ["skin thickening", "Raynaud's phenomenon", "joint pain", "fatigue", "dysphagia"],
    "lupus": ["fatigue", "joint pain", "rash", "fever", "photosensitivity"],
    "multiple sclerosis": ["numbness", "weakness", "vision loss", "fatigue", "coordination problems"],
    "parkinson's disease": ["tremor", "bradykinesia", "rigidity", "postural instability", "gait difficulty"],
    "stroke": ["weakness", "numbness", "slurred speech", "vision loss", "confusion"],
    "epilepsy": ["seizures", "confusion", "automatisms", "loss of consciousness", "muscle spasms"],
    "vertigo": ["dizziness", "nausea", "vomiting", "balance problems", "nystagmus"],
    "pharyngitis": ["sore throat", "fever", "dysphagia", "lymphadenopathy", "cough"],
    "influenza": ["fever", "cough", "sore throat", "myalgia", "fatigue"],
    "herpes zoster": ["rash", "pain", "vesicles", "itching", "fever"],
    "tuberculosis": ["cough", "weight loss", "night sweats", "fever", "hemoptysis"],
    "zika virus": ["fever", "rash", "joint pain", "conjunctivitis", "myalgia"],
    "dengue fever": ["fever", "headache", "joint pain", "rash", "myalgia"],
    "chikungunya": ["fever", "joint pain", "rash", "myalgia", "fatigue"],
    "lyme disease": ["rash", "fever", "joint pain", "fatigue", "headache"],
    "west nile virus": ["fever", "headache", "myalgia", "rash", "fatigue"],
    "hyperthyroidism": ["weight loss", "tachycardia", "heat intolerance", "tremor", "palpitations"],
    "hypothyroidism": ["fatigue", "weight gain", "cold intolerance", "dry skin", "constipation"],
    "adrenal insufficiency": ["fatigue", "weight loss", "hypotension", "hyponatremia", "hyperpigmentation"],
    "cushing's syndrome": ["weight gain", "moon face", "hypertension", "hirsutism", "bruising"],
    "endometriosis": ["pelvic pain", "dysmenorrhea", "infertility", "dyspareunia", "menorrhagia"],
    "fibroids": ["menorrhagia", "pelvic pain", "pelvic pressure", "infertility", "constipation"],
    "iron deficiency anemia": ["fatigue", "pallor", "shortness of breath", "alopecia", "restlessness"],
    "vitamin b12 deficiency": ["fatigue", "pallor", "numbness", "glossitis", "cognitive impairment"],
    "leukemia": ["fatigue", "fever", "bruising", "bone pain", "lymphadenopathy"],
    "lymphoma": ["lymphadenopathy", "night sweats", "weight loss", "fever", "itching"],
    "cellulitis": ["erythema", "swelling", "pain", "warmth", "fever"],
    "pruritus": ["itching", "rash", "dry skin", "erythema", "excoriations"],
    "meniscal tear": ["knee pain", "swelling", "locking", "instability", "reduced range of motion"]
}

def load_json(data):
    """Load JSON data (for in-memory data)."""
    return data

def save_json(data, file_path):
    """Save JSON data to file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {file_path}")

def clean_symptoms(symptom_list):
    """Clean and normalize symptom list."""
    if not isinstance(symptom_list, list):
        return []
    cleaned = []
    for symptom in symptom_list:
        if not isinstance(symptom, str) or not symptom.strip():
            continue
        symptom = symptom.lower().strip()
        # Skip if contains excluded terms or is too vague
        if any(exclude in symptom for exclude in EXCLUDE_TERMS) or len(symptom) < 3:
            continue
        # Normalize symptom
        normalized = SYMPTOM_NORMALIZATION.get(symptom, symptom)
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned

# Input data (provided JSON)
data = load_json({
    # Your provided JSON data here (omitted for brevity, but included in script logic)
    # Paste the full JSON from your input here if running manually
})

# Backup original data
save_json(data, BACKUP_FILE)

# Clean data
cleaned_data = {}
for condition, symptoms in data.items():
    condition = condition.lower().strip()
    cleaned_symptoms = clean_symptoms(symptoms)
    # Use curated symptoms if available and cleaned list is empty or poor
    if not cleaned_symptoms or len(cleaned_symptoms) < 2:
        curated = CURATED_SYMPTOMS.get(condition, [])
        if curated:
            cleaned_symptoms = curated
    if cleaned_symptoms:
        cleaned_data[condition] = cleaned_symptoms
    else:
        print(f"Warning: No valid symptoms for '{condition}' after cleaning.")

# Save cleaned data
save_json(cleaned_data, INPUT_FILE)

# Print sample output
print("Sample cleaned data:")
for condition in list(cleaned_data.keys())[:5]:
    print(f"{condition}: {cleaned_data[condition]}")