LAB_THRESHOLDS = {
    'psa': {'threshold': 4.0, 'unit': 'ng/mL', 'cancer': 'prostate cancer'},
    'cea': {'threshold': 5.0, 'unit': 'ng/mL', 'cancer': 'colorectal cancer'},
    'ca-125': {'threshold': 35.0, 'unit': 'U/mL', 'cancer': 'ovarian cancer'},
    'ca 19-9': {'threshold': 37.0, 'unit': 'U/mL', 'cancer': 'pancreatic cancer'},
    'afp': {'threshold': 10.0, 'unit': 'ng/mL', 'cancer': 'liver cancer'},
    'wbc': {'threshold': 11000, 'unit': '/mm³', 'cancer': 'leukemia'},
    'hgb': {'threshold': 12.0, 'unit': 'g/dL', 'cancer': 'anemia-related cancer', 'condition': 'low'},
    'crp': {'threshold': 10.0, 'unit': 'mg/L', 'cancer': 'general inflammation'},
    'esr': {'threshold': 20.0, 'unit': 'mm/hr', 'cancer': 'general inflammation'},
}

# Cancer diseases relevant for entity extraction
CANCER_DISEASES = {
    'breast cancer',
    'colorectal cancer',
    'prostate cancer',
    'lung cancer',
    'ovarian cancer',
    'lymphoma',
    'leukemia',
    'pancreatic cancer',
    'liver cancer'
}