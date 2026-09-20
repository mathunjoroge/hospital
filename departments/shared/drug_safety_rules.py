"""
departments/shared/drug_safety_rules.py
─────────────────────────────────────────
Single source of truth for drug allergy cross-reactivity groups and known
drug-drug interaction warning rules across the entire hospital system.

Clinical basis: Kenya Pharmacy and Poisons Board (PPB) Essential Medicines List,
British National Formulary (BNF), and WHO Model Formulary.
"""

# Drug Allergy Cross-Reactivity Dictionary
ALLERGY_GROUPS = {
    # ~10% cross-reactivity between penicillins and cephalosporins (P1-16)
    "penicillin": [
        "amoxicillin",
        "ampicillin",
        "penicillin",
        "augmentin",
        "piperacillin",
        "amoxil",
        "cloxacillin",
        "flucloxacillin",
        # cephalosporins — cross-reactive at ~10% rate in penicillin-allergic patients
        "cephalexin",
        "cefazolin",
        "ceftriaxone",
        "cefuroxime",
        "cefdinir",
        "ceftazidime",
        "cefepime",
        "cefotaxime",
    ],
    # Carbapenems: ~1% cross-reactivity with penicillin allergy
    "carbapenem": [
        "meropenem",
        "imipenem",
        "ertapenem",
        "doripenem",
    ],
    # Sulfonamide antibiotics — cross-react with sulfonamide-containing non-antibiotics
    # including furosemide (~40% incidence in sulfa-allergic patients) (P1-16)
    "sulfa": ["bactrim", "cotrimoxazole", "sulfamethoxazole", "septrin", "sulfadiazine"],
    "sulfonamide_nonabx": [
        "furosemide",
        "hydrochlorothiazide",
        "celecoxib",
        "probenecid",
        "glipizide",
        "indapamide",
        "metolazone",
    ],
    "nsaid": [
        "ibuprofen",
        "diclofenac",
        "naproxen",
        "aspirin",
        "indomethacin",
        "brufen",
        "piroxicam",
        "meloxicam",
        "ketoprofen",
        "mefenamic acid",
    ],
    "macrolide": ["azithromycin", "erythromycin", "clarithromycin", "roxithromycin"],
    "fluoroquinolone": [
        "ciprofloxacin",
        "levofloxacin",
        "moxifloxacin",
        "norfloxacin",
        "ofloxacin",
    ],
    "tetracycline": [
        "doxycycline",
        "tetracycline",
        "minocycline",
    ],
    "aminoglycoside": [
        "gentamicin",
        "amikacin",
        "tobramycin",
        "streptomycin",
        "kanamycin",
    ],
    "opiate": [
        "morphine",
        "pethidine",
        "codeine",
        "tramadol",
        "fentanyl",
        "oxycodone",
        "hydrocodone",
    ],
    "benzodiazepine": [
        "diazepam",
        "lorazepam",
        "midazolam",
        "alprazolam",
        "clonazepam",
        "nitrazepam",
    ],
    "statin": [
        "atorvastatin",
        "simvastatin",
        "rosuvastatin",
        "pravastatin",
        "lovastatin",
    ],
}

# Drug-Drug Interaction Warning Rules (pairs -> severity, warning message)
# P1-16 cross-sensitivity rules are listed first with their clinical basis.
CROSS_SENSITIVITY_RULES = [
    # Penicillin -> Cephalosporins: ~10% cross-reactivity
    {
        "allergen_group": "penicillin",
        "cross_reactive_group": "cephalosporin",
        "cross_reactive_drugs": [
            "cephalexin", "cefazolin", "ceftriaxone", "cefuroxime",
            "cefdinir", "ceftazidime", "cefepime", "cefotaxime",
        ],
        "severity": "HIGH",
        "incidence_pct": 10,
        "message": (
            "Patient has penicillin allergy. Cephalosporins share the beta-lactam ring "
            "and carry ~10% cross-reactivity risk. Consider alternative antibiotic class "
            "or perform supervised challenge if cephalosporin is required."
        ),
    },
    # Penicillin -> Carbapenems: ~1% cross-reactivity
    {
        "allergen_group": "penicillin",
        "cross_reactive_group": "carbapenem",
        "cross_reactive_drugs": ["meropenem", "imipenem", "ertapenem", "doripenem"],
        "severity": "MODERATE",
        "incidence_pct": 1,
        "message": (
            "Patient has penicillin allergy. Carbapenems share the beta-lactam ring "
            "but carry a much lower estimated ~1% cross-reactivity risk. "
            "Use with caution if no alternative is available."
        ),
    },
    # Sulfonamide antibiotics -> Furosemide (sulfonamide-containing diuretic)
    {
        "allergen_group": "sulfa",
        "cross_reactive_group": "sulfonamide_nonabx",
        "cross_reactive_drugs": [
            "furosemide", "hydrochlorothiazide", "celecoxib",
            "probenecid", "glipizide", "indapamide", "metolazone",
        ],
        "severity": "MODERATE",
        "incidence_pct": 40,
        "message": (
            "Patient has sulfonamide antibiotic allergy. Furosemide and other "
            "sulfonamide-containing non-antibiotics carry an estimated 40% "
            "cross-reactivity risk. Review allergy history before prescribing."
        ),
    },
    # NSAIDs -> Aspirin: aspirin is in the NSAID group; flag explicit co-prescription
    {
        "allergen_group": "nsaid",
        "cross_reactive_group": "nsaid",
        "cross_reactive_drugs": ["aspirin"],
        "severity": "HIGH",
        "incidence_pct": 100,
        "message": (
            "Patient has NSAID allergy. Aspirin is an NSAID and is contraindicated. "
            "Use paracetamol for analgesia if an alternative is needed."
        ),
    },
    # Fluoroquinolone class cross-reactivity
    {
        "allergen_group": "fluoroquinolone",
        "cross_reactive_group": "fluoroquinolone",
        "cross_reactive_drugs": ["ciprofloxacin", "levofloxacin", "moxifloxacin", "norfloxacin", "ofloxacin"],
        "severity": "HIGH",
        "incidence_pct": 70,
        "message": (
            "Patient has fluoroquinolone allergy. All fluoroquinolones share "
            "structural similarities and significant cross-reactivity (~70%) is expected. "
            "Avoid all fluoroquinolone antibiotics."
        ),
    },
    # Aminoglycoside cross-reactivity (ototoxicity/nephrotoxicity synergism)
    {
        "allergen_group": "aminoglycoside",
        "cross_reactive_group": "aminoglycoside",
        "cross_reactive_drugs": ["gentamicin", "amikacin", "tobramycin", "streptomycin", "kanamycin"],
        "severity": "HIGH",
        "incidence_pct": 80,
        "message": (
            "Patient has aminoglycoside allergy. High cross-reactivity exists within "
            "the aminoglycoside class due to shared structural features. "
            "Choose a non-aminoglycoside antibiotic."
        ),
    },
]

KNOWN_INTERACTIONS = [
    # ── Anticoagulant / Antiplatelet Interactions ──────────────────────────
    (
        {"warfarin", "aspirin"},
        "CRITICAL",
        "High risk of major gastrointestinal hemorrhage and severe bleeding.",
    ),
    (
        {"warfarin", "ibuprofen"},
        "HIGH",
        "Increased risk of bleeding and gastric mucosal ulceration.",
    ),
    (
        {"warfarin", "ciprofloxacin"},
        "HIGH",
        "Ciprofloxacin inhibits warfarin metabolism (CYP1A2), increasing INR and bleeding risk. Monitor INR closely.",
    ),
    (
        {"warfarin", "metronidazole"},
        "HIGH",
        "Metronidazole inhibits warfarin metabolism. INR can increase dramatically; monitor closely.",
    ),
    (
        {"warfarin", "fluconazole"},
        "HIGH",
        "Fluconazole (CYP2C9 inhibitor) significantly potentiates warfarin. INR monitoring is mandatory.",
    ),
    (
        {"warfarin", "amiodarone"},
        "CRITICAL",
        "Amiodarone markedly potentiates warfarin anticoagulation. Risk of life-threatening haemorrhage.",
    ),
    (
        {"warfarin", "paracetamol"},
        "MODERATE",
        "Regular high-dose paracetamol (>2g/day) may modestly increase INR. Monitor anticoagulation status.",
    ),
    # ── Cardiovascular Interactions ────────────────────────────────────────
    (
        {"lisinopril", "spironolactone"},
        "HIGH",
        "Severe hyperkalemia risk; requires close serum potassium monitoring.",
    ),
    (
        {"ace inhibitors", "nsaids"},
        "MODERATE",
        "NSAIDs reduce the antihypertensive effect of ACE inhibitors and increase risk of acute kidney injury.",
    ),
    (
        {"ace inhibitors", "potassium"},
        "HIGH",
        "ACE inhibitors reduce urinary potassium excretion. Co-prescription with potassium supplements risks hyperkalaemia.",
    ),
    (
        {"digoxin", "amiodarone"},
        "CRITICAL",
        "Amiodarone inhibits digoxin renal excretion, raising digoxin levels to toxic range. Reduce digoxin dose by 50% and monitor closely.",
    ),
    (
        {"digoxin", "furosemide"},
        "HIGH",
        "Furosemide-induced hypokalaemia increases the risk of digoxin toxicity and arrhythmias.",
    ),
    (
        {"amiodarone", "simvastatin"},
        "HIGH",
        "Amiodarone inhibits CYP3A4, increasing simvastatin exposure and rhabdomyolysis risk. Limit simvastatin to 20mg/day.",
    ),
    # ── QT-Prolonging Drug Combinations ───────────────────────────────────
    (
        {"ciprofloxacin", "amiodarone"},
        "HIGH",
        "Both prolong the QT interval. Combined use significantly increases risk of Torsades de Pointes and sudden cardiac death.",
    ),
    (
        {"clarithromycin", "amiodarone"},
        "HIGH",
        "Combined QT prolongation risk. Avoid this combination; monitor ECG if unavoidable.",
    ),
    (
        {"haloperidol", "ciprofloxacin"},
        "HIGH",
        "Both prolong the QT interval; risk of fatal arrhythmia. Choose an alternative antibiotic.",
    ),
    (
        {"methadone", "ciprofloxacin"},
        "HIGH",
        "Combined QT-prolonging effect significantly increases Torsades de Pointes risk.",
    ),
    # ── CYP3A4 / Drug Metabolism Interactions ─────────────────────────────
    (
        {"ciprofloxacin", "antacid"},
        "MEDIUM",
        "Chelation reduces ciprofloxacin bioavailability and therapeutic efficacy. Separate doses by at least 2 hours.",
    ),
    (
        {"simvastatin", "clarithromycin"},
        "HIGH",
        "Clarithromycin (CYP3A4 inhibitor) greatly increases simvastatin AUC; severe rhabdomyolysis risk. Use alternative statin.",
    ),
    (
        {"simvastatin", "erythromycin"},
        "HIGH",
        "Erythromycin (CYP3A4 inhibitor) increases simvastatin levels; rhabdomyolysis risk.",
    ),
    (
        {"atorvastatin", "clarithromycin"},
        "HIGH",
        "Clarithromycin (CYP3A4 inhibitor) raises atorvastatin levels; risk of rhabdomyolysis.",
    ),
    (
        {"simvastatin", "fluconazole"},
        "HIGH",
        "Fluconazole (CYP3A4/CYP2C9 inhibitor) markedly increases statin exposure and myopathy risk.",
    ),
    (
        {"rifampicin", "warfarin"},
        "CRITICAL",
        "Rifampicin strongly induces CYP2C9, drastically reducing warfarin efficacy. INR will fall; large dose adjustments needed.",
    ),
    (
        {"rifampicin", "hormonal contraceptive"},
        "HIGH",
        "Rifampicin is a potent enzyme inducer that reduces hormonal contraceptive efficacy. Use additional contraception.",
    ),
    (
        {"rifampicin", "antiretroviral"},
        "HIGH",
        "Rifampicin induces CYP3A4 and reduces plasma levels of most antiretrovirals. Consult HIV specialist for dose adjustment.",
    ),
    # ── Renal Failure / Nephrotoxicity ────────────────────────────────────
    (
        {"metformin", "contrast"},
        "HIGH",
        "Risk of contrast-induced acute renal failure and metformin lactic acidosis. Withhold metformin 48h before and after iodinated contrast.",
    ),
    (
        {"metformin", "contrast dye"},
        "HIGH",
        "Risk of lactic acidosis.",
    ),
    (
        {"nsaids", "aminoglycoside"},
        "HIGH",
        "Combined nephrotoxicity risk. NSAIDs reduce renal perfusion, increasing aminoglycoside-induced tubular toxicity.",
    ),
    (
        {"gentamicin", "furosemide"},
        "HIGH",
        "Furosemide enhances gentamicin nephrotoxicity and ototoxicity. Monitor renal function and drug levels.",
    ),
    (
        {"nsaids", "diuretics"},
        "MODERATE",
        "NSAIDs antagonise diuretic effect and increase risk of acute kidney injury, particularly in elderly or volume-depleted patients.",
    ),
    # ── Serotonin Syndrome Combinations ───────────────────────────────────
    (
        {"ssris", "tramadol"},
        "HIGH",
        "High risk of serotonin syndrome. Clinical features include agitation, tachycardia, hyperthermia, and myoclonus.",
    ),
    (
        {"ssris", "linezolid"},
        "CRITICAL",
        "Linezolid is a weak MAO inhibitor. Combination with SSRIs carries a significant risk of life-threatening serotonin syndrome.",
    ),
    (
        {"ssris", "nsaids"},
        "MODERATE",
        "Increased risk of GI bleeding. SSRIs reduce platelet aggregation; NSAIDs compound gastrointestinal injury.",
    ),
    (
        {"ssris", "triptans"},
        "MODERATE",
        "Possible serotonin syndrome with combined use. Monitor for signs of excess serotonergic activity.",
    ),
    # ── Diabetes / Glycaemia Interactions ─────────────────────────────────
    (
        {"metformin", "alcohol"},
        "MODERATE",
        "Alcohol potentiates the risk of metformin-associated lactic acidosis, particularly in hepatic impairment.",
    ),
    (
        {"glibenclamide", "fluconazole"},
        "HIGH",
        "Fluconazole (CYP2C9 inhibitor) increases sulfonylurea plasma levels, causing prolonged and severe hypoglycaemia.",
    ),
    (
        {"insulin", "beta-blockers"},
        "MODERATE",
        "Beta-blockers mask tachycardia (a key early hypoglycaemia warning sign) and can prolong insulin-induced hypoglycaemia.",
    ),
    # ── Diuretics / Electrolyte Interactions ──────────────────────────────
    (
        {"diuretics", "lithium"},
        "MODERATE",
        "Thiazide and loop diuretics reduce lithium renal clearance, increasing plasma lithium and risk of toxicity.",
    ),
    (
        {"furosemide", "aminoglycoside"},
        "HIGH",
        "Combined ototoxicity and nephrotoxicity risk. Both agents are individually toxic to the renal tubules and cochlea.",
    ),
    # ── Respiratory / Asthma Interactions ─────────────────────────────────
    (
        {"beta-blockers", "salbutamol"},
        "MODERATE",
        "Non-selective beta-blockers antagonise beta-2-mediated bronchodilation, potentially precipitating severe bronchospasm.",
    ),
    (
        {"nsaids", "asthma"},
        "HIGH",
        "NSAIDs (especially aspirin) can precipitate life-threatening bronchospasm in aspirin-sensitive asthma (~10% of asthmatics).",
    ),
    # ── HIV/TB Specific Interactions ──────────────────────────────────────
    (
        {"efavirenz", "rifampicin"},
        "HIGH",
        "Rifampicin significantly reduces efavirenz plasma concentration. Increase efavirenz dose to 800mg/day or use rifabutin.",
    ),
    (
        {"nevirapine", "fluconazole"},
        "MODERATE",
        "Fluconazole increases nevirapine exposure, potentially increasing hepatotoxicity risk.",
    ),
    # ── Miscellaneous Critical Pairs ──────────────────────────────────────
    (
        {"methotrexate", "nsaids"},
        "CRITICAL",
        "NSAIDs reduce methotrexate renal clearance, raising plasma levels to potentially fatal concentrations. Avoid concurrent use.",
    ),
    (
        {"methotrexate", "trimethoprim"},
        "CRITICAL",
        "Both drugs inhibit dihydrofolate reductase; additive myelosuppression and mucositis risk. Contraindicated combination.",
    ),
    (
        {"phenytoin", "valproate"},
        "HIGH",
        "Valproate displaces phenytoin from plasma proteins and inhibits its metabolism, causing phenytoin toxicity despite normal total plasma levels.",
    ),
    (
        {"carbamazepine", "isoniazid"},
        "HIGH",
        "Isoniazid inhibits carbamazepine metabolism (CYP3A4), causing dose-dependent carbamazepine toxicity (diplopia, ataxia, drowsiness).",
    ),
    (
        {"clozapine", "ciprofloxacin"},
        "HIGH",
        "Ciprofloxacin inhibits CYP1A2, markedly raising clozapine plasma levels and increasing risk of seizures and agranulocytosis.",
    ),
]

