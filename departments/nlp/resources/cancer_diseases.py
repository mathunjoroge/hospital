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
CANCER_TERMS = [
    # General cancer-related terms
    'tumor', 'cancer', 'malignancy', 'malignant', 'benign', 'carcinoma',
    'sarcoma', 'adenocarcinoma', 'lymphoma', 'leukemia', 'metastasis', 'metastatic',
    'oncology', 'oncologist', 'neoplasm', 'neoplastic', 'dysplasia', 'hyperplasia',
    'anaplasia', 'carcinogenesis', 'tumorigenesis', 'oncogenesis', 'mass', 'lesion',
    'polyp', 'adenoma', 'in situ carcinoma', 'invasive carcinoma', 'primary tumor',
    'secondary tumor', 'cancer stage', 'cancer grade', 'prognosis', 'relapse', 'remission',

    # Breast cancer-specific terms
    'breast lump', 'breast mass', 'nipple retraction', 'nipple discharge',
    'ductal carcinoma', 'lobular carcinoma', 'triple negative', 'HER2 positive',
    'ER positive', 'PR positive', 'inflammatory breast cancer', 'Paget disease of the breast',
    'phyllodes tumor', 'angiosarcoma of the breast', 'invasive ductal carcinoma',
    'invasive lobular carcinoma', 'DCIS', 'LCIS', 'breast cancer recurrence',
    'axillary lymph node', 'sentinel lymph node', 'breast calcifications',

    # Genetic and molecular biomarkers
    'BRCA', 'BRCA1', 'BRCA2', 'TP53', 'EGFR', 'KRAS', 'ALK', 'BRAF',
    'HER2', 'ER', 'PR', 'p53', 'microsatellite instability', 'mismatch repair',
    'tumor markers', 'CEA', 'CA-125', 'CA 15-3', 'CA 19-9', 'PSA', 'AFP', 'LDH',
    'NTRK', 'PIK3CA', 'PTEN', 'RB1', 'RAS', 'RET', 'MET', 'ROS1', 'PD-L1',
    'MSI-H', 'TMB', 'tumor mutational burden', 'genomic profiling', 'liquid biopsy',
    'circulating tumor DNA', 'ctDNA', 'oncogene', 'tumor suppressor gene', 'epigenetics',

    # Diagnostic procedures and imaging
    'biopsy', 'fine needle aspiration', 'core needle biopsy', 'excisional biopsy',
    'incisional biopsy', 'PET scan', 'CT scan', 'MRI', 'ultrasound', 'mammogram',
    'pap smear', 'colonoscopy', 'bone scan', 'tumor grading', 'tumor staging',
    'TNM staging', 'frozen section', 'histology', 'pathology', 'cytology',
    'endoscopy', 'bronchoscopy', 'sigmoidoscopy', 'esophagogastroduodenoscopy',
    'digital rectal exam', 'bone marrow biopsy', 'lumbar puncture', 'flow cytometry',
    'immunohistochemistry', 'FISH testing', 'next-generation sequencing', 'X-ray',
    'PET-CT', 'SPECT scan', 'contrast-enhanced imaging', 'molecular imaging',

    # Treatment modalities
    'chemotherapy', 'radiation therapy', 'radiotherapy', 'brachytherapy',
    'immunotherapy', 'targeted therapy', 'hormonal therapy', 'checkpoint inhibitor',
    'surgery', 'lumpectomy', 'mastectomy', 'neoadjuvant therapy', 'adjuvant therapy',
    'palliative care', 'stem cell transplant', 'bone marrow transplant',
    'proton therapy', 'cryotherapy', 'ablation therapy', 'radiofrequency ablation',
    'hyperthermic intraperitoneal chemotherapy', 'HIPEC', 'photodynamic therapy',
    'monoclonal antibodies', 'CAR-T therapy', 'bispecific antibodies', 'anti-angiogenic therapy',
    'endocrine therapy', 'parp inhibitors', 'tyrosine kinase inhibitors', 'mTOR inhibitors',
    'palliative radiotherapy', 'stereotactic radiosurgery', 'gamma knife', 'cyberknife',

    # Hematologic cancers
    'acute lymphoblastic leukemia', 'acute myeloid leukemia', 'chronic lymphocytic leukemia',
    'chronic myeloid leukemia', 'hodgkin lymphoma', 'non-hodgkin lymphoma', 'multiple myeloma',
    'myelodysplastic syndrome', 'polycythemia vera', 'essential thrombocythemia',
    'myelofibrosis', 'T-cell lymphoma', 'B-cell lymphoma', 'mantle cell lymphoma',
    'follicular lymphoma', 'diffuse large B-cell lymphoma', 'Burkitt lymphoma',
    'plasma cell neoplasm', 'amyloidosis', 'leukemic phase', 'blast crisis',

    # Gynecological cancers
    'cervical cancer', 'ovarian cancer', 'endometrial cancer', 'uterine cancer',
    'vaginal cancer', 'vulvar cancer', 'fallopian tube cancer', 'gestational trophoblastic disease',
    'choriocarcinoma', 'serous carcinoma', 'endometrioid carcinoma', 'clear cell carcinoma',
    'mucinous carcinoma', 'CA-125', 'human papillomavirus', 'HPV', 'cervical dysplasia',

    # Gastrointestinal cancers
    'colorectal cancer', 'colon cancer', 'rectal cancer', 'gastric cancer',
    'stomach cancer', 'pancreatic cancer', 'hepatocellular carcinoma', 'esophageal cancer',
    'anal cancer', 'gallbladder cancer', 'cholangiocarcinoma', 'small intestine cancer',
    'GIST', 'gastrointestinal stromal tumor', 'neuroendocrine tumor', 'carcinoid tumor',
    'Krukenberg tumor', 'Barrett esophagus', 'H. pylori infection',

    # Genitourinary cancers
    'prostate cancer', 'PSA', 'bladder cancer', 'renal cell carcinoma', 'testicular cancer',
    'penile cancer', 'urethral cancer', 'Wilms tumor', 'germ cell tumor', 'seminoma',
    'non-seminomatous germ cell tumor', 'prostate-specific antigen', 'transitional cell carcinoma',

    # Head and neck cancers
    'nasopharyngeal carcinoma', 'laryngeal cancer', 'oral cancer', 'thyroid cancer',
    'salivary gland cancer', 'oropharyngeal cancer', 'hypopharyngeal cancer',
    'paranasal sinus cancer', 'papillary thyroid carcinoma', 'follicular thyroid carcinoma',
    'medullary thyroid carcinoma', 'anaplastic thyroid carcinoma', 'squamous cell carcinoma of the head and neck',

    # Lung and thoracic cancers
    'lung cancer', 'small cell lung cancer', 'non-small cell lung cancer', 'mesothelioma',
    'thymic carcinoma', 'pulmonary carcinoid', 'adenosquamous carcinoma', 'large cell carcinoma',
    'bronchioloalveolar carcinoma', 'pleural effusion', 'pancoast tumor',

    # Skin and soft tissue cancers
    'melanoma', 'basal cell carcinoma', 'squamous cell carcinoma', 'skin cancer',
    'Kaposi sarcoma', 'Merkel cell carcinoma', 'dermatofibrosarcoma protuberans',
    'actinic keratosis', 'Bowen disease', 'malignant melanoma', 'lentigo maligna',

    # Pediatric cancers
    'neuroblastoma', 'retinoblastoma', 'Wilms tumor', 'medulloblastoma', 'Ewing sarcoma',
    'osteosarcoma', 'rhabdomyosarcoma', 'hepatoblastoma', 'germ cell tumor',
    'juvenile myelomonocytic leukemia', 'craniopharyngioma', 'pineoblastoma',

    # Brain and central nervous system cancers
    'glioblastoma', 'astrocytoma', 'oligodendroglioma', 'CNS tumor', 'brain metastases',
    'meningioma', 'ependymoma', 'glioma', 'pituitary adenoma', 'schwannoma',
    'neurofibroma', 'craniopharyngioma', 'medulloblastoma', 'pineal tumor', 'choroid plexus tumor',

    # Metastasis and advanced disease
    'liver metastasis', 'bone metastasis', 'brain metastasis', 'lung metastasis',
    'peritoneal carcinomatosis', 'pleural metastasis', 'adrenal metastasis',
    'lymph node metastasis', 'distant metastasis', 'oligometastatic disease',
    'systemic metastasis', 'regional metastasis',

    # Cancer-related symptoms
    'unexplained weight loss', 'fatigue', 'night sweats', 'lymphadenopathy',
    'anemia', 'bleeding', 'painful mass', 'jaundice', 'hematuria', 'hemoptysis',
    'dyspnea', 'ascites', 'pleural effusion', 'bone pain', 'fracture', 'edema',
    'anorexia', 'cachexia', 'paraneoplastic syndrome', 'hypercalcemia', 'pruritus',
    'cough', 'dysphagia', 'hoarseness', 'bowel obstruction', 'urinary obstruction',

    # Screening and prevention
    'cancer screening', 'mammography', 'PSA testing', 'colonoscopy', 'fecal occult blood test',
    'pap smear', 'HPV testing', 'low-dose CT scan', 'BRCA testing', 'genetic screening',
    'stool DNA test', 'virtual colonoscopy', 'breast self-exam', 'clinical breast exam',
    'skin self-exam', 'oral cancer screening', 'lung cancer screening', 'chemoprevention',
    'prophylactic mastectomy', 'prophylactic oophorectomy', 'cancer vaccine',

    # Other cancer-related terms
    'second primary cancer', 'recurrence', 'complete remission', 'partial remission',
    'progressive disease', 'stable disease', 'tumor burden', 'cancer survivor',
    'palliative chemotherapy', 'clinical trial', 'precision medicine', 'cancer immunotherapy',
    'tumor microenvironment', 'angiogenesis', 'apoptosis', 'cancer stem cell',
    'tumor heterogeneity', 'cancer cachexia', 'oncologic emergency', 'febrile neutropenia',
    'tumor lysis syndrome', 'superior vena cava syndrome', 'spinal cord compression'
]
CANCER_PATTERNS = {
    'BREAST_CANCER': [
        ('SYMPTOM', r'\b(palpable lump|breast lump|nipple retraction|nipple discharge|breast mass|skin dimpling|breast pain|swelling in breast)\b'),
        ('DIAGNOSIS', r'\b(ductal carcinoma|lobular carcinoma|brca|her2 positive|triple negative breast cancer|invasive carcinoma|paget disease)\b')
    ],
    'COLON_CANCER': [
        ('SYMPTOM', r'\b(blood in stool|rectal bleeding|abdominal pain|change in bowel habits|unexplained weight loss|constipation|diarrhea|iron deficiency anemia)\b'),
        ('DIAGNOSIS', r'\b(colorectal cancer|adenocarcinoma|polyps|familial adenomatous polyposis|lynch syndrome|microsatellite instability)\b')
    ],
    'PROSTATE_CANCER': [
        ('SYMPTOM', r'\b(urinary hesitancy|frequent urination|nocturia|difficulty urinating|pelvic pain|blood in urine|weak urine stream)\b'),
        ('DIAGNOSIS', r'\b(prostate cancer|adenocarcinoma of prostate|elevated psa|gleason score|bone metastasis)\b')
    ],
    'LUNG_CANCER': [
        ('SYMPTOM', r'\b(persistent cough|hemoptysis|chest pain|shortness of breath|wheezing|hoarseness|recurrent pneumonia)\b'),
        ('DIAGNOSIS', r'\b(non-small cell lung cancer|small cell lung cancer|squamous cell carcinoma|adenocarcinoma of lung|egfr mutation|alk rearrangement)\b')
    ],
    'OVARIAN_CANCER': [
        ('SYMPTOM', r'\b(abdominal bloating|pelvic pain|early satiety|urinary urgency|abdominal distension|fatigue|nausea)\b'),
        ('DIAGNOSIS', r'\b(serous carcinoma|mucinous carcinoma|endometrioid carcinoma|clear cell carcinoma|ca-125 elevation|ovarian mass|brca mutation)\b')
    ],
    'PANCREATIC_CANCER': [
        ('SYMPTOM', r'\b(jaundice|weight loss|abdominal pain|back pain|dark urine|clay-colored stool|nausea|loss of appetite)\b'),
        ('DIAGNOSIS', r'\b(pancreatic adenocarcinoma|pancreatic tumor|ca 19-9|pancreatitis history|ductal carcinoma of pancreas)\b')
    ],
    'LIVER_CANCER': [
        ('SYMPTOM', r'\b(right upper quadrant pain|abdominal swelling|jaundice|fatigue|weight loss|nausea|ascites)\b'),
        ('DIAGNOSIS', r'\b(hepatocellular carcinoma|hcc|liver mass|afp elevation|cirrhosis|hbv|hcv)\b')
    ],
    'LEUKEMIA': [
        ('SYMPTOM', r'\b(fatigue|easy bruising|frequent infections|fever|bone pain|pallor|bleeding gums|night sweats)\b'),
        ('DIAGNOSIS', r'\b(acute myeloid leukemia|chronic lymphocytic leukemia|acute lymphoblastic leukemia|cml|aml|abnormal wbc count)\b')
    ],
    'LYMPHOMA': [
        ('SYMPTOM', r'\b(painless lymphadenopathy|night sweats|fever|unexplained weight loss|fatigue|itching)\b'),
        ('DIAGNOSIS', r'\b(hodgkin lymphoma|non-hodgkin lymphoma|reed-sternberg cells|b-cell lymphoma|t-cell lymphoma)\b')
    ],
    'CERVICAL_CANCER': [
        ('SYMPTOM', r'\b(postcoital bleeding|pelvic pain|foul vaginal discharge|intermenstrual bleeding|pain during sex)\b'),
        ('DIAGNOSIS', r'\b(hpv|cervical intraepithelial neoplasia|squamous cell carcinoma of cervix|adenocarcinoma of cervix|pap smear abnormality)\b')
    ],
    'ESOPHAGEAL_CANCER': [
        ('SYMPTOM', r'\b(dysphagia|odynophagia|weight loss|chest pain|hoarseness|regurgitation|heartburn)\b'),
        ('DIAGNOSIS', r'\b(squamous cell carcinoma of esophagus|adenocarcinoma of esophagus|barrett esophagus|esophageal mass)\b')
    ],
    'STOMACH_CANCER': [
        ('SYMPTOM', r'\b(abdominal pain|early satiety|nausea|vomiting|weight loss|black stools|anorexia)\b'),
        ('DIAGNOSIS', r'\b(gastric carcinoma|adenocarcinoma of stomach|h. pylori|intestinal metaplasia|linitis plastica)\b')
    ],
    'KIDNEY_CANCER': [
        ('SYMPTOM', r'\b(blood in urine|flank pain|abdominal mass|weight loss|fatigue|fever|night sweats)\b'),
        ('DIAGNOSIS', r'\b(renal cell carcinoma|clear cell carcinoma|wilms tumor|renal mass|von hippel-lindau)\b')
    ],
    'BLADDER_CANCER': [
        ('SYMPTOM', r'\b(painless hematuria|frequent urination|dysuria|urinary urgency|lower abdominal pain)\b'),
        ('DIAGNOSIS', r'\b(transitional cell carcinoma|urothelial carcinoma|bladder mass|cytology abnormality|cystoscopy finding)\b')
    ],
    'THYROID_CANCER': [
        ('SYMPTOM', r'\b(neck lump|hoarseness|difficulty swallowing|neck swelling|pain in throat|persistent cough)\b'),
        ('DIAGNOSIS', r'\b(papillary carcinoma|follicular carcinoma|medullary thyroid cancer|anaplastic carcinoma|thyroid nodule)\b')
    ],
    'SKIN_CANCER': [
        ('SYMPTOM', r'\b(changing mole|skin lesion|non-healing ulcer|itchy mole|bleeding lesion|dark spot)\b'),
        ('DIAGNOSIS', r'\b(melanoma|basal cell carcinoma|squamous cell carcinoma|skin biopsy|dysplastic nevus)\b')
    ],
    'BRAIN_CANCER': [
        ('SYMPTOM', r'\b(headache|seizures|vision changes|nausea|vomiting|personality changes|speech difficulty)\b'),
        ('DIAGNOSIS', r'\b(glioblastoma|astrocytoma|meningioma|medulloblastoma|brain tumor|cns lesion)\b')
    ],
    'MULTIPLE_MYELOMA': [
        ('SYMPTOM', r'\b(bone pain|fatigue|frequent infections|anemia|back pain|weakness)\b'),
        ('DIAGNOSIS', r'\b(monoclonal spike|m protein|plasma cell neoplasm|bence jones protein|lytic lesions)\b')
    ],
    'SARCOMA': [
        ('SYMPTOM', r'\b(painless mass|swelling|limited range of motion|bone pain|fatigue)\b'),
        ('DIAGNOSIS', r'\b(osteosarcoma|ewing sarcoma|leiomyosarcoma|liposarcoma|soft tissue sarcoma)\b')
    ]
}



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


# ~/projects/hospital/departments/nlp/resources/cancer_diseases.py
import os

CANCER_KEYWORDS_FILE = os.path.join(os.path.dirname(__file__), "cancer_keywords.json")