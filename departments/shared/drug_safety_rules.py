"""
departments/shared/drug_safety_rules.py
─────────────────────────────────────────
Single source of truth for drug allergy cross-reactivity groups and known
drug-drug interaction warning rules across the entire hospital system.
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
        # cephalosporins — cross-reactive at ~10% rate in penicillin-allergic patients
        "cephalexin",
        "cefazolin",
        "ceftriaxone",
        "cefuroxime",
        "cefdinir",
        "ceftazidime",
    ],
    # Sulfonamide antibiotics — cross-react with sulfonamide-containing non-antibiotics
    # including furosemide (~40% incidence in sulfa-allergic patients) (P1-16)
    "sulfa": ["bactrim", "cotrimoxazole", "sulfamethoxazole", "septrin"],
    "sulfonamide_nonabx": [
        "furosemide",
        "hydrochlorothiazide",
        "celecoxib",
        "probenecid",
        "glipizide",
    ],
    "nsaid": [
        "ibuprofen",
        "diclofenac",
        "naproxen",
        "aspirin",
        "indomethacin",
        "brufen",
    ],
    "macrolide": ["azithromycin", "erythromycin", "clarithromycin"],
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
            "cefdinir", "ceftazidime",
        ],
        "severity": "HIGH",
        "incidence_pct": 10,
        "message": (
            "Patient has penicillin allergy. Cephalosporins share the beta-lactam ring "
            "and carry ~10% cross-reactivity risk. Consider alternative antibiotic class "
            "or perform supervised challenge if cephalosporin is required."
        ),
    },
    # Sulfonamide antibiotics -> Furosemide (sulfonamide-containing diuretic)
    {
        "allergen_group": "sulfa",
        "cross_reactive_group": "sulfonamide_nonabx",
        "cross_reactive_drugs": [
            "furosemide", "hydrochlorothiazide", "celecoxib",
            "probenecid", "glipizide",
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
]

KNOWN_INTERACTIONS = [
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
        {"lisinopril", "spironolactone"},
        "HIGH",
        "Severe hyperkalemia risk; requires close serum potassium monitoring.",
    ),
    (
        {"ciprofloxacin", "antacid"},
        "MEDIUM",
        "Chelation reduces ciprofloxacin bioavailability and therapeutic efficacy.",
    ),
    (
        {"metformin", "contrast"},
        "HIGH",
        "Risk of contrast-induced acute renal failure and metformin lactic acidosis.",
    ),
    (
        {"simvastatin", "clarithromycin"},
        "HIGH",
        "Risk of rhabdomyolysis.",
    ),
    (
        {"simvastatin", "erythromycin"},
        "HIGH",
        "Risk of rhabdomyolysis.",
    ),
    (
        {"atorvastatin", "clarithromycin"},
        "HIGH",
        "Risk of rhabdomyolysis.",
    ),
    (
        {"metformin", "contrast dye"},
        "HIGH",
        "Risk of lactic acidosis.",
    ),
    (
        {"ace inhibitors", "nsaids"},
        "MODERATE",
        "May reduce kidney function.",
    ),
    (
        {"diuretics", "lithium"},
        "MODERATE",
        "Risk of lithium toxicity.",
    ),
    (
        {"ssris", "nsaids"},
        "MODERATE",
        "Increased risk of bleeding.",
    ),
]
