# Fallback dictionary mapping common disease names to UMLS Concept Unique Identifiers (CUIs)
fallback_disease_keywords = {
    "pneumonia": "C0032285",
    "community-acquired pneumonia": "C0032285",
    "hospital-acquired pneumonia": "C0032285",
    "lung infection": "C0032285",
    "malaria": "C0024530",
    "falciparum malaria": "C0024530",
    "vivax malaria": "C0024530",
    "typhoid": "C0041466",
    "enteric fever": "C0041466",
    "dengue": "C0011311",
    "tuberculosis": "C0041296",
    "active tuberculosis": "C0041296",
    "latent tuberculosis": "C0041296",
    "meningitis": "C0025289",
    "bacterial meningitis": "C0025289",
    "viral meningitis": "C0025289",
    "urinary tract infection": "C0033578",
    "uti": "C0033578",
    "gastroenteritis": "C0017160",
    "infectious diarrhea": "C0017160",
    "hepatitis": "C0019158",
    "viral hepatitis": "C0019158",
    "hepatitis b": "C0019159",
    "hepatitis c": "C0019160",
    "influenza": "C0021400",
    "covid-19": "C5203670",
    "stroke": "C0038454",
    "ischemic stroke": "C0038454",
    "hemorrhagic stroke": "C0038454",
    "heart attack": "C0027051",
    "myocardial infarction": "C0027051",
    "diabetes": "C0011849",
    "type 1 diabetes": "C0011858",
    "type 2 diabetes": "C0011860",
    "hypertension": "C0020538",
    "essential hypertension": "C0020538",
    "bronchitis": "C0006277",
    "acute bronchitis": "C0006277",
    "asthma": "C0004096",
    "anemia": "C0002871",
    "iron deficiency anemia": "C0002871",
    "sepsis": "C0243026",
    "septic shock": "C0243026",
    "appendicitis": "C0003615",
    "acute appendicitis": "C0003615",
    "back pain": "C0004604",
    "lower back pain": "C0004604",
    "depression": "C0011570",
    "major depressive disorder": "C0011570",
    "anxiety": "C0003467",
    "generalized anxiety disorder": "C0003467",
    "jaundice": "C0022346"
}

# Mapping of symptoms to UMLS CUIs
fallback_symptom_cuis = {
    "fever": "C0015967",
    "high temperature": "C0015967",
    "cough": "C0010200",
    "dry cough": "C0010200",
    "productive cough": "C0010200",
    "headache": "C0018681",
    "nausea": "C0027497",
    "vomiting": "C0042963",
    "abdominal pain": "C0000737",
    "diarrhea": "C0011991",
    "constipation": "C0009806",
    "chest pain": "C0008031",
    "angina": "C0008031",
    "shortness of breath": "C0013404",
    "dyspnea": "C0013404",
    "dizziness": "C0012833",
    "vertigo": "C0012833",
    "fatigue": "C0015672",
    "rash": "C0037284",
    "skin rash": "C0037284",
    "loss of appetite": "C0003123",
    "anorexia": "C0003123",
    "sore throat": "C0242429",
    "pharyngitis": "C0242429",
    "joint pain": "C0003862",
    "arthralgia": "C0003862",
    "muscle pain": "C0026858",
    "myalgia": "C0026858",
    "urinary frequency": "C0235526",
    "polyuria": "C0235526",
    "dysuria": "C0013428",
    "painful urination": "C0013428",
    "hematuria": "C0018965",
    "blood in urine": "C0018965",
    "jaundice": "C0022346",
    "yellow eyes": "C0022346",
    "cyanosis": "C0010520"
}

# Management Plans (Updated with Latest Guidelines as of 2024–2025)
fallback_management_plans = {
    "pneumonia": (
        "Administer appropriate antibiotics based on likely pathogens (e.g., amoxicillin, azithromycin, or ceftriaxone). "
        "Supportive care includes oxygen therapy if SpO₂ <94%, hydration, antipyretics. Monitor respiratory status, vitals, and response to treatment. "
        "Consider hospitalization for CURB-65 score ≥2 or severe symptoms."
    ),
    "malaria": (
        "Initiate artemisinin-based combination therapy (ACT), e.g., artemether-lumefantrine. "
        "Monitor for complications like cerebral malaria, hypoglycemia, renal failure. "
        "Provide paracetamol for fever, ensure adequate hydration. Refer to higher center if severe."
    ),
    "dengue": (
        "Fluid resuscitation guided by vital signs and hematocrit levels. Avoid NSAIDs due to bleeding risk. "
        "Monitor platelet count and watch for warning signs: persistent vomiting, fluid accumulation, altered sensorium. "
        "Hospitalize if plasma leakage or hemorrhage suspected."
    ),
    "typhoid": (
        "Start empirical fluoroquinolones (if susceptible) or third-generation cephalosporins (e.g., ceftriaxone). "
        "Ensure hydration, rest, and nutrition. Consider typhoid conjugate vaccine for prevention in endemic areas."
    ),
    "tuberculosis": (
        "Begin RIPE regimen (Rifampin, Isoniazid, Pyrazinamide, Ethambutol) for drug-susceptible TB. "
        "Ensure directly observed therapy (DOT), monitor liver enzymes, report to public health authorities. "
        "Drug susceptibility testing is critical for MDR-TB cases."
    ),
    "meningitis": (
        "Start empiric IV antibiotics (e.g., ceftriaxone + vancomycin ± ampicillin). Perform lumbar puncture if safe. "
        "Monitor for raised intracranial pressure, seizures, sepsis. Dexamethasone may be added in bacterial meningitis."
    ),
    "gastroenteritis": (
        "Oral rehydration with ORS; IV fluids if dehydrated. Zinc supplements for children under 5. "
        "Avoid routine antibiotics unless dysentery or high suspicion of bacterial cause. Symptomatic relief with loperamide (adults only)."
    ),
    "covid-19": (
        "Isolation, mask use, and symptom monitoring. For moderate to severe: remdesivir, dexamethasone, supplemental oxygen. "
        "Anticoagulation considered for hospitalized patients. Monoclonal antibodies or nirmatrelvir/ritonavir for high-risk outpatients."
    ),
    "stroke": (
        "Urgent neuroimaging (CT or MRI), BP control (<185/105 mmHg for thrombolysis candidates). "
        "Thrombolysis within 4.5 hours or mechanical thrombectomy within 24 hours (based on imaging eligibility). "
        "Aspirin initiated after 24 hrs if not contraindicated."
    ),
    "myocardial infarction": (
        "Immediate aspirin, nitroglycerin for chest pain, and ECG. Activate cath lab for STEMI. "
        "Dual antiplatelet therapy (aspirin + clopidogrel/ticagrelor), heparin, beta-blockers, statins. "
        "Reperfusion via PCI or fibrinolytic therapy if indicated."
    ),
    "uti": (
        "First-line: Nitrofurantoin or trimethoprim-sulfamethasoxazole. "
        "For complicated UTI or pyelonephritis: ceftriaxone or oral fluoroquinolones. "
        "Increase fluid intake and manage pain/symptoms."
    ),
    "hepatitis": (
        "Rest, avoid alcohol and hepatotoxic drugs. Monitor LFTs. Antiviral therapy for HBV/HCV. "
        "Vaccinate for Hepatitis A/B. Manage complications like ascites or encephalopathy if present."
    ),
    "influenza": (
        "Oseltamivir within 48 hours of onset for high-risk patients. Rest, hydration, isolation. "
        "Annual vaccination recommended. Supportive care with antipyretics and analgesics."
    ),
    "bronchitis": (
        "Symptomatic management: hydration, humidified air, cough suppressants. "
        "Bronchodilators if wheezing. Antibiotics not routinely recommended unless secondary bacterial infection suspected."
    ),
    "hypertension": (
        "Lifestyle modifications: low sodium diet, weight loss, exercise. "
        "First-line meds: thiazide diuretics, ACE inhibitors, calcium channel blockers. "
        "Monitor regularly; escalate therapy if BP remains uncontrolled."
    ),
    "diabetes": (
        "Type 1: Insulin therapy (basal-bolus or pump). Type 2: Metformin first-line, SGLT2 inhibitors preferred for CV benefit. "
        "Monitor HbA1c, educate on foot care, eye exams, and lifestyle changes."
    ),
    "asthma": (
        "Stepwise approach: short-acting beta-agonists (SABA) for rescue, inhaled corticosteroids (ICS), long-acting beta agonists (LABA) if needed. "
        "Biologics for severe asthma. Educate on triggers and peak flow monitoring."
    ),
    "anemia": (
        "Investigate underlying cause: iron studies, B12/folate levels. Iron supplementation for IDA, transfusion if symptomatic. "
        "EPO for CKD-related anemia, refer for hematology evaluation if chronic or complex."
    ),
    "back pain": (
        "NSAIDs, muscle relaxants, physical therapy. Assess red flags: fever, weight loss, neurological deficits. "
        "Imaging reserved for red flag presence or non-response to conservative management."
    ),
    "sepsis": (
        "Early recognition using qSOFA or NEWS. Start broad-spectrum antibiotics within 1 hour, give IV fluids, vasopressors if hypotensive. "
        "Monitor lactate, urine output, and organ function. Early goal-directed therapy (EGDT) principles applied."
    ),
    "depression": (
        "Psychotherapy (CBT, IPT), SSRIs/SNRIs for moderate to severe depression. "
        "Monitor for suicidal ideation, side effects. Combine with lifestyle interventions and social support."
    ),
    "anxiety": (
        "Cognitive behavioral therapy (CBT), mindfulness techniques, breathing exercises. "
        "SSRIs/SNRIs for persistent symptoms. Short-term benzodiazepines for acute panic attacks (with caution)."
    )
}

# Common Symptom-Disease Associations
COMMON_SYMPTOM_DISEASE_MAP = {
    "fever": [
        "Malaria", "Influenza", "Pneumonia", "COVID-19", 
        "Urinary Tract Infection", "Sepsis", "Tuberculosis", "Typhoid Fever"
    ],
    "chills": [
        "Malaria", "Influenza", "Pneumonia", "Sepsis",
        "Bacterial Infection", "Viral Infection", "Pyelonephritis"
    ],
    "anorexia": [
        "Depression", "Cancer", "HIV/AIDS", "Chronic Kidney Disease",
        "Hepatitis", "Inflammatory Bowel Disease", "Anorexia Nervosa"
    ],
    "nausea": [
        "Gastroenteritis", "Migraine", "Food Poisoning", 
        "Pregnancy", "Pancreatitis", "Peptic Ulcer", "Hyperemesis Gravidarum"
    ],
    "headache": [
        "Migraine", "Tension Headache", "Sinusitis", 
        "Hypertension", "Meningitis", "Subarachnoid Hemorrhage"
    ]
}
SYMPTOM_NORMALIZATIONS =  {
        # Fever-related
        "fever": "fever",
        "fevers": "fever",
        "febrile": "fever",
        "pyrexia": "fever",
        "high temperature": "fever",
        "temperature": "fever",

        # Chills
        "chill": "chills",
        "chills": "chills",
        "rigor": "chills",
        "rigors": "chills",
        "shivering": "chills",

        # Cough
        "cough": "cough",
        "coughing": "cough",
        "dry cough": "cough",
        "productive cough": "cough",
        "wet cough": "cough",

        # Shortness of breath
        "shortness of breath": "shortness of breath",
        "dyspnea": "shortness of breath",
        "difficulty breathing": "shortness of breath",
        "breathless": "shortness of breath",
        "breathlessness": "shortness of breath",

        # Chest pain
        "chest pain": "chest pain",
        "chest discomfort": "chest pain",
        "chest ache": "chest pain",
        "cardiac pain": "chest pain",

        # Nausea
        "nausea": "nausea",
        "nauseated": "nausea",
        "sick to stomach": "nausea",
        "queasy": "nausea",
        "upset stomach": "nausea",

        # Headache
        "headache": "headache",
        "head pain": "headache",
        "cephalgia": "headache",
        "migraine": "headache",  # Could also be separate if needed
        "tension headache": "headache",

        # Weakness
        "weakness": "weakness",
        "weak": "weakness",
        "fatigue": "weakness",
        "lack of strength": "weakness",
        "tiredness": "weakness",

        # Visual disturbances
        "visual disturbance": "visual disturbances",
        "vision changes": "visual disturbances",
        "blurred vision": "visual disturbances",
        "double vision": "visual disturbances",
        "loss of vision": "visual disturbances",

        # Slurred speech
        "slurred speech": "slurred speech",
        "speech difficulty": "slurred speech",
        "dysarthria": "slurred speech",
        "trouble speaking": "slurred speech",

        # Abdominal pain
        "abdominal pain": "abdominal pain",
        "stomach pain": "abdominal pain",
        "belly pain": "abdominal pain",
        "abdominal ache": "abdominal pain",
        "tummy ache": "abdominal pain",

        # Diarrhea
        "diarrhea": "diarrhea",
        "loose stools": "diarrhea",
        "frequent loose stools": "diarrhea",
        "watery stool": "diarrhea",

        # Vomiting
        "vomiting": "vomiting",
        "vomit": "vomiting",
        "throw up": "vomiting",
        "emesis": "vomiting",

        # Loss of appetite
        "loss of appetite": "loss of appetite",
        "decreased appetite": "loss of appetite",
        "poor appetite": "loss of appetite",
        "appetite loss": "loss of appetite",
        "anorexia": "loss of appetite",
        "not eating well": "loss of appetite",

        # Urinary frequency
        "urinary frequency": "urinary frequency",
        "frequent urination": "urinary frequency",
        "peeing more": "urinary frequency",
        "polyuria": "urinary frequency",

        # Dysuria
        "dysuria": "dysuria",
        "painful urination": "dysuria",
        "burning when peeing": "dysuria",
        "urinary pain": "dysuria",

        # Joint pain
        "joint pain": "joint pain",
        "arthralgia": "joint pain",
        "joint ache": "joint pain",
        "joint soreness": "joint pain",

        # Fatigue
        "fatigue": "fatigue",
        "tired": "fatigue",
        "exhaustion": "fatigue",
        "lethargy": "fatigue",

        # Sore throat
        "sore throat": "sore throat",
        "throat pain": "sore throat",
        "pharyngitis": "sore throat",
        "sore neck": "sore throat",

        # Polyuria
        "polyuria": "polyuria",
        "excessive urination": "polyuria",
        "too much urine": "polyuria",

        # Thirst
        "thirst": "thirst",
        "excessive thirst": "thirst",
        "increased thirst": "thirst",
        "polydipsia": "thirst",

        # Jaundice
        "jaundice": "jaundice",
        "yellow eyes": "jaundice",
        "yellow skin": "jaundice",
        "icterus": "jaundice",

        # Weight loss
        "weight loss": "weight loss",
        "unintentional weight loss": "weight loss",
        "losing weight": "weight loss",
        "thin": "weight loss",

        # Palpitations
        "palpitations": "palpitations",
        "heart palpitations": "palpitations",
        "racing heart": "palpitations",
        "heart racing": "palpitations",

        # Swelling
        "swelling": "swelling",
        "edema": "swelling",
        "fluid retention": "swelling",
        "puffiness": "swelling",

        # Back pain
        "back pain": "back pain",
        "low back pain": "back pain",
        "lumbar pain": "back pain",
        "backache": "back pain",

        # Malaise
        "malaise": "malaise",
        "general ill feeling": "malaise",
        "just not feeling right": "malaise",
        "unwell": "malaise",

        # Rash
        "rash": "rash",
        "skin rash": "rash",
        "eruption": "rash",
        "skin irritation": "rash"
    }