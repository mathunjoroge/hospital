# default_patterns.py

DEFAULT_PATTERNS = [
    ("PAIN", r"\b(pain|ache|discomfort|tenderness|soreness|burning|stabbing)\b"),
    ("FEVER", r"\b(fever|pyrexia|hyperthermia|febrile|temperature of \d{2,3})\b"),
    ("RESPIRATORY", r"\b(cough|dyspnea|shortness of breath|wheezing|tachypnea|hemoptysis|stridor|respiratory distress)\b"),
    ("CARDIO", r"\b(chest pain|palpitations?|tachycardia|bradycardia|syncope|fainting|orthopnea|PND|paroxysmal nocturnal dyspnea)\b"),
    ("GASTRO", r"\b(nausea|vomiting|diarrhea|constipation|abdominal pain|bloody stool|hematemesis|melena|dyspepsia|bloating)\b"),
    ("NEURO", r"\b(headache|dizziness|vertigo|confusion|seizures?|weakness|numbness|tingling|aphasia|ataxia|paresthesia|tremor|syncope|LOC|loss of consciousness)\b"),
    ("BACK_PAIN", r"\b(back pain|backache|lumbago|spinal pain)\b"),
    ("CHILLS", r"\b(chills|shivering|rigors)\b"),
    ("APPETITE_LOSS", r"\b(loss of appetite|anorexia|reduced appetite)\b"),
    ("JAUNDICE", r"\b(jaundice|yellowing of eyes|yellow skin|icterus)\b"),
    ("URINARY", r"\b(dysuria|hematuria|frequency|urgency|nocturia|incontinence|retention|painful urination|burning urination)\b"),
    ("OB_GYN", r"\b(vaginal bleeding|pelvic pain|amenorrhea|menorrhagia|spotting|cramps|fetal movement|contractions|labor pain)\b"),
    ("MUSCULOSKELETAL", r"\b(joint pain|myalgia|arthralgia|swelling|stiffness|muscle weakness|limited range of motion)\b"),
    ("SKIN", r"\b(rash|itching|pruritus|hives|urticaria|blister|lesion|skin discoloration)\b"),
    ("MENTAL", r"\b(depression|anxiety|suicidal thoughts|hallucinations|psychosis|agitation|delirium)\b"),
    ("BLEEDING", r"\b(bleeding|hemorrhage|blood loss|epistaxis|hematemesis|melena|hematochezia)\b"),
    ("VISION", r"\b(blurred vision|vision loss|diplopia|photophobia|eye pain)\b"),
    ("HEARING", r"\b(hearing loss|ear pain|tinnitus|ear discharge)\b"),
    ("THROAT", r"\b(sore throat|pharyngitis|odynophagia|difficulty swallowing|hoarseness)\b"),
    ("GENERAL", r"\b(fatigue|malaise|unwell|lethargy|tiredness|weakness|weight loss)\b")
]
