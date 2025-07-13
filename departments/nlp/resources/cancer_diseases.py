cancer_diseases = {
    'prostate cancer': {
        'cui': 'C0376358',
        'symptoms': {'weight loss', 'fatigue', 'pelvic pain', 'urinary urgency', 'hematuria', 'dysuria', 'erectile dysfunction', 'bone pain'},
        'risk_factors': {'age >50', 'family history', 'African descent', 'BRCA mutations', 'obesity'},
        'diagnostic_methods': {'PSA test', 'digital rectal exam', 'transrectal ultrasound', 'mpMRI', 'biopsy', 'bone scan'},
        'treatments': {'active surveillance', 'radical prostatectomy', 'radiation therapy', 'brachytherapy', 'ADT', 'chemotherapy'},
        'staging': ['TNM system', 'Gleason score'],
        'epidemiology': {
            'prevalence': 'Most common male cancer (US)',
            'median_age': 66,
            '5yr_survival': '~99% (localized)'
        }
    },
    
    'lymphoma': {
        'cui': 'C0024299',
        'symptoms': {'night sweats', 'weight loss', 'fatigue', 'lymphadenopathy', 'fever', 'pruritus', 'splenomegaly', 'chest pain'},
        'subtypes': {
            'Hodgkin': {'Reed-Sternberg cells', 'EBV association'},
            'Non-Hodgkin': {'DLBCL', 'follicular lymphoma', 'mantle cell'}
        },
        'risk_factors': {'immunosuppression', 'EBV infection', 'HIV', 'autoimmune diseases', 'pesticides'},
        'diagnostic_methods': {'excisional biopsy', 'PET-CT', 'bone marrow biopsy', 'flow cytometry', 'LDH test'},
        'treatments': {'chemotherapy (ABVD/R-CHOP)', 'immunotherapy', 'radiation', 'stem cell transplant', 'CAR-T therapy'},
        'emerging_targets': {'CD30', 'PD-1/PD-L1'}
    },
    
    'leukemia': {
        'cui': 'C0023418',
        'symptoms': {'fatigue', 'weight loss', 'fever', 'easy bruising', 'recurrent infections', 'bone pain', 'pallor', 'splenomegaly'},
        'classification': {
            'acute': {'AML': 'myeloid', 'ALL': 'lymphoid'},
            'chronic': {'CML': 'Philadelphia chromosome', 'CLL': 'CD5+ B-cells'}
        },
        'risk_factors': {'benzene exposure', 'previous chemotherapy', 'Down syndrome', 'ionizing radiation', 'genetic syndromes'},
        'diagnostic_methods': {'CBC with differential', 'bone marrow biopsy', 'flow cytometry', 'cytogenetics', 'FISH testing'},
        'treatments': {'chemotherapy', 'targeted therapy (TKIs)', 'stem cell transplant', 'immunotherapy', 'CAR-T cells'},
        'prognostic_markers': {'FLT3 mutations', 'Philadelphia chromosome', 'TP53 status'}
    },
    
    'lung cancer': {
        'cui': 'C0242379',
        'symptoms': {'cough', 'weight loss', 'chest pain', 'hemoptysis', 'dyspnea', 'hoarseness', 'bone pain', 'superior vena cava syndrome'},
        'types': {
            'NSCLC': {'adenocarcinoma', 'squamous cell', 'large cell'},
            'SCLC': {'rapid growth', 'neuroendocrine features'}
        },
        'risk_factors': {'tobacco smoking', 'radon exposure', 'asbestos', 'air pollution', 'family history'},
        'diagnostic_methods': {'LDCT screening', 'bronchoscopy', 'PET-CT', 'transthoracic biopsy', 'liquid biopsy'},
        'treatments': {'lobectomy', 'SBRT', 'chemotherapy', 'immunotherapy', 'targeted therapy (EGFR/ALK)'},
        'screening_criteria': {'55-80 yrs', '30 pack-year history'}
    },
    
    'colorectal cancer': {
        'cui': 'C0009402',
        'symptoms': {'abdominal pain', 'weight loss', 'rectal bleeding', 'change in bowel habits', 'iron-deficiency anemia', 'tenesmus', 'incomplete evacuation'},
        'molecular_pathways': {'chromosomal instability', 'microsatellite instability', 'CpG island methylation'},
        'risk_factors': {'inflammatory bowel disease', 'familial adenomatous polyposis', 'Lynch syndrome', 'red meat consumption', 'sedentary lifestyle'},
        'diagnostic_methods': {'colonoscopy', 'FIT test', 'CT colonography', 'CEA monitoring', 'KRAS/NRAS testing'},
        'treatments': {'colectomy', 'neoadjuvant chemoradiation', 'FOLFOX/FOLFIRI', 'anti-EGFR therapy', 'immunotherapy for MSI-H'},
        'prevention': {'aspirin chemoprevention', 'polyp removal'}
    },
    
    'ovarian cancer': {
        'cui': 'C0029925',
        'symptoms': {'abdominal bloating', 'weight loss', 'pelvic pain', 'early satiety', 'urinary frequency', 'back pain', 'constipation'},
        'types': {'epithelial (90%)', 'germ cell', 'stromal'},
        'risk_factors': {'BRCA mutations', 'Lynch syndrome', 'nulliparity', 'endometriosis', 'talc use'},
        'diagnostic_methods': {'transvaginal ultrasound', 'CA-125', 'CT/MRI', 'paracentesis', 'genetic testing'},
        'treatments': {'debulking surgery', 'platinum-based chemo', 'PARP inhibitors', 'anti-angiogenics'},
        'screening_controversy': 'No effective screening for general population'
    },
    
    'pancreatic cancer': {
        'cui': 'C0235974',
        'symptoms': {'weight loss', 'jaundice', 'abdominal pain', 'Courvoisier sign', 'new-onset diabetes', 'steatorrhea', 'Trousseau sign'},
        'anatomical_sites': {'head (75%)', 'body', 'tail'},
        'risk_factors': {'smoking', 'chronic pancreatitis', 'diabetes', 'BRCA2 mutations', 'familial atypical mole syndrome'},
        'diagnostic_methods': {'contrast CT', 'ERCP', 'EUS with biopsy', 'CA19-9', 'molecular profiling'},
        'treatments': {'Whipple procedure', 'FOLFIRINOX', 'gemcitabine/nab-paclitaxel', 'palliative stenting'},
        'prognosis': {'5yr_survival': '<10%', 'reasons': 'late presentation, aggressive biology'}
    },
    
    'liver cancer': {
        'cui': 'C2239176',
        'symptoms': {'weight loss', 'jaundice', 'abdominal pain', 'ascites', 'hepatomegaly', 'caput medusae', 'hepatic encephalopathy'},
        'types': {'HCC (90%)', 'cholangiocarcinoma', 'angiosarcoma'},
        'risk_factors': {'hepatitis B/C', 'alcoholic cirrhosis', 'NAFLD', 'aflatoxin exposure', 'hemochromatosis'},
        'diagnostic_methods': {'multiphase CT/MRI', 'AFP monitoring', 'biopsy (controversial)', 'LIRADS classification'},
        'treatments': {'ablation', 'TACE', 'sorafenib', 'liver transplant', 'immunotherapy'},
        'surveillance_protocol': {'Ultrasound + AFP q6mo for cirrhotics'}
    },
    
    'breast cancer': {
        'cui': 'C0006142',
        'symptoms': {'breast lump', 'weight loss', 'nipple discharge', 'skin dimpling', 'nipple retraction', 'peau d\'orange', 'axillary lymphadenopathy'},
        'molecular_subtypes': {'HR+/HER2-', 'HER2+', 'triple negative'},
        'risk_factors': {'BRCA mutations', 'early menarche', 'nulliparity', 'HRT use', 'chest radiation'},
        'diagnostic_methods': {'mammography', 'tomosynthesis', 'breast MRI', 'core biopsy', 'Oncotype DX'},
        'treatments': {'lumpectomy', 'mastectomy', 'radiation', 'endocrine therapy', 'HER2-targeted agents'},
        'prevention': {'tamoxifen/raloxifene', 'prophylactic mastectomy'}
    },
    
    # Additional cancers
    'melanoma': {
        'cui': 'C0025202',
        'symptoms': {'changing mole', 'new pigmented lesion', 'pruritus', 'bleeding', 'asymmetry/border irregularity'},
        'risk_factors': {'UV exposure', 'dysplastic nevus syndrome', 'fair skin', 'xeroderma pigmentosum'},
        'staging': 'Breslow thickness, ulceration, mitotic rate',
        'treatments': {'wide excision', 'sentinel lymph node biopsy', 'immunotherapy', 'targeted therapy (BRAF/MEK)'},
        'prevention': {'ABCDE rule', 'sun protection'}
    },
    
    'glioblastoma': {
        'cui': 'C0017636',
        'symptoms': {'headache', 'seizures', 'neurological deficits', 'nausea', 'personality changes'},
        'molecular_features': {'IDH-wildtype', 'MGMT methylation', 'EGFR amplification'},
        'treatments': {'maximal safe resection', 'temozolomide', 'TTFields', 'bevacizumab'},
        'prognosis': {'median_survival': '15 months'}
    },
    
    'cervical cancer': {
        'cui': 'C0007862',
        'symptoms': {'postcoital bleeding', 'abnormal discharge', 'pelvic pain', 'dysuria'},
        'pathogenesis': {'HPV integration (types 16/18)', 'cervical intraepithelial neoplasia'},
        'prevention': {'HPV vaccination', 'Pap smear screening'},
        'treatments': {'LEEP', 'radical hysterectomy', 'chemoradiation'},
        'global_impact': 'Leading cause of cancer death in low-resource settings'
    }
}
cancer_symptoms = {
    # Original symptoms
    'unexplained weight loss', 'persistent fatigue', 'night sweats', 'persistent cough',
    'palpable lump', 'abnormal bleeding', 'chronic pain', 'hoarseness', 'dysphagia',
    'breast lump', 'nipple retraction', 'nipple discharge', 'breast mass',
    
    # New symptoms
    # Gastrointestinal
    'change in bowel habits', 'blood in stool', 'rectal bleeding', 'persistent indigestion',
    'persistent abdominal pain', 'bloating',
    
    # Urinary/Reproductive
    'blood in urine', 'dysuria', 'urinary frequency', 'testicular mass',
    'postmenopausal bleeding', 'unusual vaginal discharge',
    
    # Dermatological
    'non-healing sore', 'skin ulcer', 'changes in mole', 'jaundice',
    'skin thickening', 'palmar erythema',
    
    # Respiratory/Cardiovascular
    'hemoptysis', 'shortness of breath', 'chest pain',
    
    # Neurological
    'persistent headaches', 'seizures', 'vision changes', 'hearing loss', 'balance problems',
    
    # Constitutional
    'unexplained fever', 'loss of appetite', 'early satiety',
    
    # Specific cancer markers
    'ascites', 'bone pain', 'lymphadenopathy', 'neurological deficit'
}
BREAST_CANCER_TERMS = [
    'breast lump',
    'nipple retraction',
    'nipple discharge',
    'breast mass',
    'ductal carcinoma',
    'brca'
]
BREAST_CANCER_PATTERNS = [
    ('SYMPTOM', r'\b(palpable lump|breast lump|nipple retraction|nipple discharge|breast mass)\b'),
    ('DIAGNOSIS', r'\b(ductal carcinoma|lobular carcinoma|brca)\b')
]
# Breast cancer-specific keywords
BREAST_CANCER_KEYWORDS = [
    'breast lump',
    'nipple retraction',
    'nipple discharge',
    'breast mass',
    'ductal carcinoma',
    'brca'
]

# Breast cancer-specific keyword to CUI mappings
BREAST_CANCER_KEYWORD_CUIS = {
    'breast lump': 'C0234450',
    'nipple retraction': 'C0234451',
    'nipple discharge': 'C0027408',
    'breast mass': 'C0234450',
    'ductal carcinoma': 'C0007124',
    'brca': 'C0599878'
}
# resources/cancer_diseases.py
BREAST_CANCER_SYMPTOMS = {
    'breast lump': 'C0234450',
    'nipple retraction': 'C0234451',
    'nipple discharge': 'C0027408',
    'breast mass': 'C0234450'
}
# resources/cancer_diseases.py
CANCER_PLANS = {
    'prostate cancer': {
        'plan': 'Refer to oncologist; order prostate biopsy',
        'lab_tests': [{'test': 'PSA follow-up', 'description': 'Monitor PSA levels in 4-6 weeks'}],
        'cancer_follow_up': 'Consider imaging (e.g., CT/MRI) and biopsy if indicated.'
    },
    'lymphoma': {
        'plan': 'Order lymph node biopsy; consider PET scan',
        'lab_tests': [{'test': 'LDH', 'description': 'Assess lymphoma activity'}],
        'cancer_follow_up': 'Monitor symptoms; consider tumor marker tests.'
    },
    'leukemia': {
        'plan': 'Refer to hematologist; order bone marrow biopsy',
        'lab_tests': [{'test': 'CBC follow-up', 'description': 'Monitor WBC and other blood counts'}],
        'cancer_follow_up': 'Consider cytogenetic testing.'
    },
    'lung cancer': {
        'plan': 'Refer to oncologist; order chest CT and biopsy',
        'lab_tests': [{'test': 'Sputum cytology', 'description': 'Assess for malignant cells'}],
        'cancer_follow_up': 'Consider PET scan for staging.'
    },
    'colorectal cancer': {
        'plan': 'Refer to oncologist; order colonoscopy and biopsy',
        'lab_tests': [{'test': 'CEA', 'description': 'Monitor colorectal cancer markers'}],
        'cancer_follow_up': 'Consider CT abdomen/pelvis.'
    },
    'ovarian cancer': {
        'plan': 'Refer to oncologist; order pelvic ultrasound and biopsy',
        'lab_tests': [{'test': 'CA-125', 'description': 'Monitor ovarian cancer markers'}],
        'cancer_follow_up': 'Consider CT/MRI for staging.'
    },
    'pancreatic cancer': {
        'plan': 'Refer to oncologist; order abdominal CT and biopsy',
        'lab_tests': [{'test': 'CA 19-9', 'description': 'Monitor pancreatic cancer markers'}],
        'cancer_follow_up': 'Consider endoscopic ultrasound.'
    },
    'liver cancer': {
        'plan': 'Refer to oncologist; order liver ultrasound and biopsy',
        'lab_tests': [{'test': 'AFP', 'description': 'Monitor liver cancer markers'}],
        'cancer_follow_up': 'Consider MRI liver.'
    },
    'breast cancer': {
        'plan': 'Refer to oncologist; order mammogram and biopsy',
        'lab_tests': [{'test': 'BRCA testing', 'description': 'Assess genetic risk'}],
        'cancer_follow_up': 'Consider breast MRI.'
    }
}