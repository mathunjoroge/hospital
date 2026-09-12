"""
departments/shared/drug_safety_rules.py
─────────────────────────────────────────
Single source of truth for drug allergy cross-reactivity groups and known
drug-drug interaction warning rules across the entire hospital system.
"""

# Drug Allergy Cross-Reactivity Dictionary
ALLERGY_GROUPS = {
    "penicillin": [
        "amoxicillin",
        "ampicillin",
        "penicillin",
        "augmentin",
        "piperacillin",
        "amoxil",
    ],
    "sulfa": ["bactrim", "cotrimoxazole", "sulfamethoxazole", "septrin"],
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
