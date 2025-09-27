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
    # General/Constitutional
    'unexplained weight loss', 'persistent fatigue', 'night sweats', 'unexplained fever',
    'loss of appetite', 'early satiety', 'cachexia', 'generalized weakness', 'malaise',
    'chills', 'anorexia', 'pallor', 'pruritus',

    # Gastrointestinal
    'change in bowel habits', 'blood in stool', 'rectal bleeding', 'persistent indigestion',
    'persistent abdominal pain', 'bloating', 'tenesmus', 'narrow stools', 'clay-colored stools',
    'hematemesis', 'melena', 'abdominal distension', 'nausea', 'vomiting', 'epigastric pain',
    'feeling of fullness', 'dysphagia', 'odynophagia', 'regurgitation', 'heartburn',

    # Urinary/Reproductive
    'blood in urine', 'dysuria', 'urinary frequency', 'testicular mass', 'postmenopausal bleeding',
    'unusual vaginal discharge', 'hematospermia', 'urinary hesitancy', 'weak urine stream',
    'urinary urgency', 'pelvic pain', 'painful intercourse', 'intermenstrual bleeding',
    'scrotal swelling', 'gynecomastia', 'lower abdominal pain', 'urinary retention',

    # Dermatological
    'non-healing sore', 'skin ulcer', 'changes in mole', 'jaundice', 'skin thickening',
    'palmar erythema', 'bleeding lesion', 'itchy mole', 'asymmetrical lesion', 'irregular border',
    'multicolored lesion', 'crusting lesion', 'dark spot', 'petechiae', 'purpura', 'erythema',
    'peau d’orange', 'spider angiomas', 'hyperpigmentation', 'actinic keratosis',

    # Respiratory/Cardiovascular
    'persistent cough', 'hemoptysis', 'shortness of breath', 'chest pain', 'wheezing',
    'hoarseness', 'clubbing of fingers', 'pleural effusion', 'recurrent pneumonia', 'dyspnea',
    'hiccups',

    # Neurological
    'persistent headaches', 'seizures', 'vision changes', 'hearing loss', 'balance problems',
    'personality changes', 'speech difficulty', 'motor weakness', 'cognitive decline',
    'memory loss', 'confusion', 'coordination problems', 'numbness or tingling',
    'cranial nerve deficits',

    # Musculoskeletal
    'bone pain', 'pathological fracture', 'joint pain', 'limited range of motion',
    'muscle weakness', 'swelling in limbs',

    # Hematological/Immune
    'lymphadenopathy', 'easy bruising', 'frequent infections', 'bleeding gums', 'epistaxis',
    'splenomegaly', 'hepatomegaly',

    # Breast-Specific
    'breast lump', 'nipple retraction', 'nipple discharge', 'breast mass', 'breast asymmetry',
    'breast tenderness', 'breast warmth', 'nipple inversion', 'skin dimpling', 'axillary swelling',

    # Specific Cancer Markers
    'ascites', 'neurological deficit', 'paraneoplastic syndromes', 'Horner syndrome',
    'superior vena cava syndrome', 'pericardial effusion', 'facial swelling', 'neck lump',
    'thyroid enlargement', 'soft tissue mass', 'foul-smelling discharge', 'postcoital bleeding'
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
        ('SYMPTOM', r'\b(palpable lump|breast lump|nipple retraction|nipple discharge|breast mass|skin dimpling|breast pain|swelling in breast|breast asymmetry|peau d\'orange|nipple inversion|erythema of breast|breast tenderness|axillary swelling|unexplained breast warmth)\b'),
        ('DIAGNOSIS', r'\b(ductal carcinoma|lobular carcinoma|brca1|brca2|her2 positive|triple negative breast cancer|invasive carcinoma|paget disease|inflammatory breast cancer|dcis|lcis|phyllodes tumor|mammogram abnormality|bi-rads|breast cancer metastasis)\b'),
        ('RISK_FACTOR', r'\b(family history of breast cancer|hormone replacement therapy|early menarche|late menopause|nulliparity|obesity|alcohol consumption|brca mutation)\b')
    ],
    'COLON_CANCER': [
        ('SYMPTOM', r'\b(blood in stool|rectal bleeding|abdominal pain|change in bowel habits|unexplained weight loss|constipation|diarrhea|iron deficiency anemia|narrow stools|tenesmus|abdominal distension|fatigue|cramping pain)\b'),
        ('DIAGNOSIS', r'\b(colorectal cancer|adenocarcinoma|polyps|familial adenomatous polyposis|lynch syndrome|microsatellite instability|kras mutation|cea elevation|colonoscopy finding|sigmoidoscopy abnormality|bowel obstruction)\b'),
        ('RISK_FACTOR', r'\b(family history of colon cancer|crohn disease|ulcerative colitis|high red meat diet|low fiber diet|smoking|obesity|diabetes)\b')
    ],
    'PROSTATE_CANCER': [
        ('SYMPTOM', r'\b(urinary hesitancy|frequent urination|nocturia|difficulty urinating|pelvic pain|blood in urine|weak urine stream|erectile dysfunction|painful ejaculation|lower back pain|hematospermia)\b'),
        ('DIAGNOSIS', r'\b(prostate cancer|adenocarcinoma of prostate|elevated psa|gleason score|bone metastasis|prostate-specific antigen|prostate nodule|digital rectal exam abnormality|t3 stage|t4 stage|metastatic prostate cancer)\b'),
        ('RISK_FACTOR', r'\b(family history of prostate cancer|age over 50|african ancestry|high-fat diet|brca mutation)\b')
    ],
    'LUNG_CANCER': [
        ('SYMPTOM', r'\b(persistent cough|hemoptysis|chest pain|shortness of breath|wheezing|hoarseness|recurrent pneumonia|fatigue|weight loss|clubbing of fingers|dyspnea|paraneoplastic syndrome)\b'),
        ('DIAGNOSIS', r'\b(non-small cell lung cancer|small cell lung cancer|squamous cell carcinoma|adenocarcinoma of lung|egfr mutation|alk rearrangement|ros1 mutation|kras mutation|lung nodule|bronchoscopy finding|ct chest abnormality|pet scan uptake)\b'),
        ('RISK_FACTOR', r'\b(smoking|secondhand smoke|radon exposure|asbestos exposure|family history of lung cancer|copd|pulmonary fibrosis)\b')
    ],
    'OVARIAN_CANCER': [
        ('SYMPTOM', r'\b(abdominal bloating|pelvic pain|early satiety|urinary urgency|abdominal distension|fatigue|nausea|constipation|back pain|menstrual irregularities|painful intercourse)\b'),
        ('DIAGNOSIS', r'\b(serous carcinoma|mucinous carcinoma|endometrioid carcinoma|clear cell carcinoma|ca-125 elevation|ovarian mass|brca mutation|germ cell tumor|sex cord-stromal tumor|transvaginal ultrasound abnormality|ovarian cyst)\b'),
        ('RISK_FACTOR', r'\b(family history of ovarian cancer|brca1|brca2|endometriosis|infertility|hormone therapy|obesity)\b')
    ],
    'PANCREATIC_CANCER': [
        ('SYMPTOM', r'\b(jaundice|weight loss|abdominal pain|back pain|dark urine|clay-colored stool|nausea|loss of appetite|new-onset diabetes|pruritus|fatigue|pancreatic mass)\b'),
        ('DIAGNOSIS', r'\b(pancreatic adenocarcinoma|pancreatic tumor|ca 19-9|pancreatitis history|ductal carcinoma of pancreas|neuroendocrine tumor|ipmn|mucinous cystic neoplasm|pancreatic cyst|ercp abnormality|ct pancreas finding)\b'),
        ('RISK_FACTOR', r'\b(smoking|chronic pancreatitis|family history of pancreatic cancer|diabetes|obesity|brca2 mutation)\b')
    ],
    'LIVER_CANCER': [
        ('SYMPTOM', r'\b(right upper quadrant pain|abdominal swelling|jaundice|fatigue|weight loss|nausea|ascites|anorexia|pruritus|hepatic encephalopathy|spider angiomas)\b'),
        ('DIAGNOSIS', r'\b(hepatocellular carcinoma|hcc|liver mass|afp elevation|cirrhosis|hbv|hcv|cholangiocarcinoma|angiosarcoma|fibrolamellar carcinoma|liver biopsy finding)\b'),
        ('RISK_FACTOR', r'\b(hepatitis b|hepatitis c|alcohol abuse|nonalcoholic fatty liver disease|hemochromatosis|aflatoxin exposure|family history of liver cancer)\b')
    ],
    'LEUKEMIA': [
        ('SYMPTOM', r'\b(fatigue|easy bruising|frequent infections|fever|bone pain|pallor|bleeding gums|night sweats|petechiae|splenomegaly|hepatomegaly|lymphadenopathy)\b'),
        ('DIAGNOSIS', r'\b(acute myeloid leukemia|chronic lymphocytic leukemia|acute lymphoblastic leukemia|cml|aml|abnormal wbc count|blast crisis|philadelphia chromosome|bone marrow biopsy abnormality|flow cytometry finding)\b'),
        ('RISK_FACTOR', r'\b(family history of leukemia|prior chemotherapy|radiation exposure|smoking|benzene exposure|down syndrome)\b')
    ],
    'LYMPHOMA': [
        ('SYMPTOM', r'\b(painless lymphadenopathy|night sweats|fever|unexplained weight loss|fatigue|itching|splenomegaly|hepatomegaly|chest pain|abdominal fullness)\b'),
        ('DIAGNOSIS', r'\b(hodgkin lymphoma|non-hodgkin lymphoma|reed-sternberg cells|b-cell lymphoma|t-cell lymphoma|follicular lymphoma|diffuse large b-cell lymphoma|mantle cell lymphoma|lymph node biopsy abnormality|pet scan uptake)\b'),
        ('RISK_FACTOR', r'\b(family history of lymphoma|ebv infection|hiv|autoimmune disease|organ transplant|hepatitis c)\b')
    ],
    'CERVICAL_CANCER': [
        ('SYMPTOM', r'\b(postcoital bleeding|pelvic pain|foul vaginal discharge|intermenstrual bleeding|pain during sex|abnormal vaginal bleeding|urinary symptoms|leg swelling)\b'),
        ('DIAGNOSIS', r'\b(hpv|cervical intraepithelial neoplasia|squamous cell carcinoma of cervix|adenocarcinoma of cervix|pap smear abnormality|colposcopy finding|cervical mass|cervical biopsy abnormality)\b'),
        ('RISK_FACTOR', r'\b(hpv infection|smoking|multiple sexual partners|early sexual activity|immunosuppression|hiv)\b')
    ],
    'ESOPHAGEAL_CANCER': [
        ('SYMPTOM', r'\b(dysphagia|odynophagia|weight loss|chest pain|hoarseness|regurgitation|heartburn|hiccups|vomiting|chronic cough)\b'),
        ('DIAGNOSIS', r'\b(squamous cell carcinoma of esophagus|adenocarcinoma of esophagus|barrett esophagus|esophageal mass|endoscopy finding|esophageal stricture|pet scan abnormality)\b'),
        ('RISK_FACTOR', r'\b(smoking|alcohol consumption|barrett esophagus|gerd|obesity|hpv infection)\b')
    ],
    'STOMACH_CANCER': [
        ('SYMPTOM', r'\b(abdominal pain|early satiety|nausea|vomiting|weight loss|black stools|anorexia|dysphagia|epigastric pain|hematemesis|feeling of fullness)\b'),
        ('DIAGNOSIS', r'\b(gastric carcinoma|adenocarcinoma of stomach|h. pylori|intestinal metaplasia|linitis plastica|gastric mass|endoscopy finding|signet ring cell carcinoma)\b'),
        ('RISK_FACTOR', r'\b(h. pylori infection|smoking|high salt diet|family history of stomach cancer|pernicious anemia|chronic gastritis)\b')
    ],
    'KIDNEY_CANCER': [
        ('SYMPTOM', r'\b(blood in urine|flank pain|abdominal mass|weight loss|fatigue|fever|night sweats|hypertension|edema)\b'),
        ('DIAGNOSIS', r'\b(renal cell carcinoma|clear cell carcinoma|wilms tumor|renal mass|von hippel-lindau|papillary renal cell carcinoma|chromophobe carcinoma|ct kidney abnormality)\b'),
        ('RISK_FACTOR', r'\b(smoking|obesity|hypertension|family history of kidney cancer|chronic kidney disease|von hippel-lindau syndrome)\b')
    ],
    'BLADDER_CANCER': [
        ('SYMPTOM', r'\b(painless hematuria|frequent urination|dysuria|urinary urgency|lower abdominal pain|pelvic pain|back pain|urinary retention)\b'),
        ('DIAGNOSIS', r'\b(transitional cell carcinoma|urothelial carcinoma|bladder mass|cytology abnormality|cystoscopy finding|bladder tumor recurrence)\b'),
        ('RISK_FACTOR', r'\b(smoking|chemical exposure|chronic bladder infections|schistosomiasis|family history of bladder cancer)\b')
    ],
    'THYROID_CANCER': [
        ('SYMPTOM', r'\b(neck lump|hoarseness|difficulty swallowing|neck swelling|pain in throat|persistent cough|thyroid enlargement|palpable thyroid nodule)\b'),
        ('DIAGNOSIS', r'\b(papillary carcinoma|follicular carcinoma|medullary thyroid cancer|anaplastic carcinoma|thyroid nodule|calcitonin elevation|ret mutation|thyroid ultrasound abnormality)\b'),
        ('RISK_FACTOR', r'\b(family history of thyroid cancer|radiation exposure|iodine deficiency|female gender|hashimoto thyroiditis)\b')
    ],
    'SKIN_CANCER': [
        ('SYMPTOM', r'\b(changing mole|skin lesion|non-healing ulcer|itchy mole|bleeding lesion|dark spot|asymmetrical lesion|irregular border|multicolored lesion|crusting lesion)\b'),
        ('DIAGNOSIS', r'\b(melanoma|basal cell carcinoma|squamous cell carcinoma|skin biopsy|dysplastic nevus|actinic keratosis|merkel cell carcinoma|dermoscopy finding)\b'),
        ('RISK_FACTOR', r'\b(uv exposure|fair skin|family history of skin cancer|history of sunburns|immunosuppression)\b')
    ],
    'BRAIN_CANCER': [
        ('SYMPTOM', r'\b(headache|seizures|vision changes|nausea|vomiting|personality changes|speech difficulty|motor weakness|balance issues|cognitive decline|hearing loss)\b'),
        ('DIAGNOSIS', r'\b(glioblastoma|astrocytoma|meningioma|medulloblastoma|brain tumor|cns lesion|oligodendroglioma|ependymoma|mri brain abnormality|ct brain finding)\b'),
        ('RISK_FACTOR', r'\b(radiation exposure|family history of brain cancer|neurofibromatosis|li-fraumeni syndrome)\b')
    ],
    'MULTIPLE_MYELOMA': [
        ('SYMPTOM', r'\b(bone pain|fatigue|frequent infections|anemia|back pain|weakness|hypercalcemia|renal failure|neuropathy)\b'),
        ('DIAGNOSIS', r'\b(monoclonal spike|m protein|plasma cell neoplasm|bence jones protein|lytic lesions|multiple myeloma|bone marrow biopsy abnormality|crab criteria)\b'),
        ('RISK_FACTOR', r'\b(age over 65|african ancestry|family history of multiple myeloma|mgus|obesity)\b')
    ],
    'SARCOMA': [
        ('SYMPTOM', r'\b(painless mass|swelling|limited range of motion|bone pain|fatigue|fracture|soft tissue mass)\b'),
        ('DIAGNOSIS', r'\b(osteosarcoma|ewing sarcoma|leiomyosarcoma|liposarcoma|soft tissue sarcoma|chondrosarcoma|rhabdomyosarcoma|synovial sarcoma|mri mass finding)\b'),
        ('RISK_FACTOR', r'\b(radiation exposure|li-fraumeni syndrome|paget disease|retinoblastoma history|neurofibromatosis)\b')
    ],
    'TESTICULAR_CANCER': [
        ('SYMPTOM', r'\b(testicular lump|scrotal swelling|testicular pain|heavy scrotum|abdominal pain|back pain|gynecomastia)\b'),
        ('DIAGNOSIS', r'\b(seminoma|non-seminomatous germ cell tumor|testicular mass|beta-hcg elevation|afp elevation|ultrasound scrotum abnormality)\b'),
        ('RISK_FACTOR', r'\b(cryptorchidism|family history of testicular cancer|klinefelter syndrome|testicular microlithiasis)\b')
    ],
    'ENDOMETRIAL_CANCER': [
        ('SYMPTOM', r'\b(abnormal uterine bleeding|postmenopausal bleeding|pelvic pain|heavy periods|intermenstrual bleeding|abdominal bloating)\b'),
        ('DIAGNOSIS', r'\b(endometrial adenocarcinoma|endometrioid carcinoma|uterine sarcoma|ca-125 elevation|endometrial biopsy abnormality|transvaginal ultrasound finding)\b'),
        ('RISK_FACTOR', r'\b(obesity|estrogen therapy|tamoxifen use|lynch syndrome|family history of endometrial cancer|diabetes)\b')
    ],
    'GALLBLADDER_CANCER': [
        ('SYMPTOM', r'\b(right upper quadrant pain|jaundice|weight loss|nausea|vomiting|abdominal mass|fever)\b'),
        ('DIAGNOSIS', r'\b(gallbladder carcinoma|adenocarcinoma of gallbladder|gallbladder mass|porcelain gallbladder|ct abdomen abnormality)\b'),
        ('RISK_FACTOR', r'\b(gallstones|chronic cholecystitis|obesity|female gender|family history of gallbladder cancer)\b')
    ],
    'MESOTHELIOMA': [
        ('SYMPTOM', r'\b(chest pain|shortness of breath|persistent cough|weight loss|pleural effusion|fatigue|night sweats)\b'),
        ('DIAGNOSIS', r'\b(malignant mesothelioma|pleural mesothelioma|peritoneal mesothelioma|biopsy finding|ct chest abnormality)\b'),
        ('RISK_FACTOR', r'\b(asbestos exposure|smoking|family history of mesothelioma)\b')
    ],
    'ORAL_CANCER': [
        ('SYMPTOM', r'\b(persistent mouth sore|oral ulcer|difficulty swallowing|hoarseness|mouth pain|swelling in mouth|white patches|red patches)\b'),
        ('DIAGNOSIS', r'\b(squamous cell carcinoma of mouth|oral cavity cancer|oropharyngeal cancer|hpv-related oral cancer|biopsy abnormality)\b'),
        ('RISK_FACTOR', r'\b(smoking|alcohol consumption|hpv infection|betel nut chewing|family history of oral cancer)\b')
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
        'plan': 'Refer to urologist and oncologist; order prostate biopsy; consider multiparametric MRI',
        'lab_tests': [
            {'test': 'PSA follow-up', 'description': 'Monitor PSA levels in 4-6 weeks to assess trend'},
            {'test': 'Free PSA ratio', 'description': 'Evaluate proportion of free to total PSA for diagnostic specificity'},
            {'test': 'Prostate health index (phi)', 'description': 'Assess risk of prostate cancer'},
            {'test': 'Testosterone level', 'description': 'Baseline for potential hormonal therapy'}
        ],
        'cancer_follow_up': 'Consider bone scan or PSMA PET scan for staging; evaluate Gleason score; monitor for metastasis.'
    },
    'lymphoma': {
        'plan': 'Refer to hematologist/oncologist; order excisional lymph node biopsy; consider PET/CT scan',
        'lab_tests': [
            {'test': 'LDH', 'description': 'Assess lymphoma activity and prognosis'},
            {'test': 'Beta-2 microglobulin', 'description': 'Evaluate disease burden in non-Hodgkin lymphoma'},
            {'test': 'CBC with differential', 'description': 'Monitor for anemia or abnormal lymphocyte counts'},
            {'test': 'ESR', 'description': 'Assess inflammatory activity'}
        ],
        'cancer_follow_up': 'Monitor symptoms (B symptoms: fever, night sweats, weight loss); consider flow cytometry or immunohistochemistry; stage with Ann Arbor or Lugano classification.'
    },
    'leukemia': {
        'plan': 'Refer to hematologist; order bone marrow biopsy and aspiration; consider flow cytometry',
        'lab_tests': [
            {'test': 'CBC follow-up', 'description': 'Monitor WBC, RBC, and platelet counts for disease progression'},
            {'test': 'Peripheral blood smear', 'description': 'Assess for blast cells'},
            {'test': 'Cytogenetic analysis', 'description': 'Identify chromosomal abnormalities (e.g., Philadelphia chromosome)'},
            {'test': 'LDH', 'description': 'Evaluate disease activity'}
        ],
        'cancer_follow_up': 'Consider molecular testing (e.g., BCR-ABL for CML); monitor for minimal residual disease; repeat bone marrow biopsy as needed.'
    },
    'lung cancer': {
        'plan': 'Refer to pulmonologist and oncologist; order chest CT, bronchoscopy, and biopsy; consider EBUS',
        'lab_tests': [
            {'test': 'Sputum cytology', 'description': 'Assess for malignant cells in sputum'},
            {'test': 'EGFR mutation testing', 'description': 'Identify targetable mutations for therapy'},
            {'test': 'ALK rearrangement testing', 'description': 'Assess for ALK gene alterations'},
            {'test': 'PD-L1 expression', 'description': 'Evaluate for immunotherapy eligibility'}
        ],
        'cancer_follow_up': 'Consider PET/CT for staging; monitor for paraneoplastic syndromes; evaluate TNM staging.'
    },
    'colorectal cancer': {
        'plan': 'Refer to gastroenterologist and oncologist; order colonoscopy, biopsy, and CT colonography',
        'lab_tests': [
            {'test': 'CEA', 'description': 'Monitor colorectal cancer markers for progression or recurrence'},
            {'test': 'KRAS mutation testing', 'description': 'Assess for targeted therapy eligibility'},
            {'test': 'MSI testing', 'description': 'Evaluate microsatellite instability for Lynch syndrome'},
            {'test': 'Fecal occult blood test', 'description': 'Screen for ongoing bleeding'}
        ],
        'cancer_follow_up': 'Consider CT abdomen/pelvis for staging; monitor for liver metastasis; evaluate surgical resection options.'
    },
    'ovarian cancer': {
        'plan': 'Refer to gynecologic oncologist; order transvaginal ultrasound, pelvic CT/MRI, and biopsy',
        'lab_tests': [
            {'test': 'CA-125', 'description': 'Monitor ovarian cancer markers for diagnosis and recurrence'},
            {'test': 'HE4', 'description': 'Assess additional ovarian cancer marker'},
            {'test': 'BRCA1/2 testing', 'description': 'Evaluate genetic predisposition'},
            {'test': 'Ova1 panel', 'description': 'Assess risk of malignancy in pelvic mass'}
        ],
        'cancer_follow_up': 'Consider PET/CT for staging; monitor for ascites or pleural effusion; evaluate FIGO staging.'
    },
    'pancreatic cancer': {
        'plan': 'Refer to gastroenterologist and oncologist; order abdominal CT, MRCP, and biopsy; consider EUS',
        'lab_tests': [
            {'test': 'CA 19-9', 'description': 'Monitor pancreatic cancer markers for progression'},
            {'test': 'CEA', 'description': 'Additional tumor marker for pancreatic cancer'},
            {'test': 'LFTs', 'description': 'Assess liver function for jaundice or metastasis'},
            {'test': 'BRCA2 testing', 'description': 'Evaluate genetic predisposition'}
        ],
        'cancer_follow_up': 'Consider endoscopic ultrasound for local staging; monitor for distant metastasis; evaluate Whipple procedure eligibility.'
    },
    'liver cancer': {
        'plan': 'Refer to hepatologist and oncologist; order liver ultrasound, triphasic CT/MRI, and biopsy',
        'lab_tests': [
            {'test': 'AFP', 'description': 'Monitor liver cancer markers for hepatocellular carcinoma'},
            {'test': 'LFTs', 'description': 'Assess liver function and cirrhosis'},
            {'test': 'HBV/HCV serology', 'description': 'Evaluate viral hepatitis as risk factor'},
            {'test': 'DCP (PIVKA-II)', 'description': 'Additional marker for HCC'}
        ],
        'cancer_follow_up': 'Consider MRI liver with contrast for staging; monitor for portal vein thrombosis; evaluate BCLC staging.'
    },
    'breast cancer': {
        'plan': 'Refer to oncologist and breast surgeon; order mammogram, ultrasound, and core biopsy; consider MRI',
        'lab_tests': [
            {'test': 'BRCA1/2 testing', 'description': 'Assess genetic risk for hereditary breast cancer'},
            {'test': 'HER2 testing', 'description': 'Evaluate for targeted therapy eligibility'},
            {'test': 'ER/PR status', 'description': 'Determine hormone receptor status for treatment'},
            {'test': 'Ki-67', 'description': 'Assess tumor proliferation rate'}
        ],
        'cancer_follow_up': 'Consider breast MRI or PET/CT for staging; monitor for axillary lymph node involvement; evaluate Oncotype DX score.'
    },
    'cervical cancer': {
        'plan': 'Refer to gynecologic oncologist; order colposcopy, cervical biopsy, and pelvic MRI',
        'lab_tests': [
            {'test': 'HPV testing', 'description': 'Assess high-risk HPV strains'},
            {'test': 'Pap smear follow-up', 'description': 'Monitor for persistent abnormalities'},
            {'test': 'CBC', 'description': 'Evaluate for anemia due to bleeding'},
            {'test': 'SCC antigen', 'description': 'Monitor squamous cell carcinoma progression'}
        ],
        'cancer_follow_up': 'Consider PET/CT for staging; monitor for lymph node metastasis; evaluate FIGO staging.'
    },
    'esophageal cancer': {
        'plan': 'Refer to gastroenterologist and oncologist; order upper endoscopy, biopsy, and chest/abdominal CT',
        'lab_tests': [
            {'test': 'CEA', 'description': 'Monitor tumor marker for esophageal cancer'},
            {'test': 'HER2 testing', 'description': 'Assess for targeted therapy in adenocarcinoma'},
            {'test': 'LFTs', 'description': 'Evaluate for liver metastasis'},
            {'test': 'Albumin', 'description': 'Assess nutritional status'}
        ],
        'cancer_follow_up': 'Consider EUS for local staging; monitor for dysphagia progression; evaluate TNM staging.'
    },
    'stomach cancer': {
        'plan': 'Refer to gastroenterologist and oncologist; order upper endoscopy, biopsy, and abdominal CT',
        'lab_tests': [
            {'test': 'CEA', 'description': 'Monitor gastric cancer markers'},
            {'test': 'CA 72-4', 'description': 'Additional tumor marker for gastric cancer'},
            {'test': 'H. pylori testing', 'description': 'Assess for associated infection'},
            {'test': 'CBC', 'description': 'Evaluate for anemia due to bleeding'}
        ],
        'cancer_follow_up': 'Consider PET/CT for staging; monitor for peritoneal metastasis; evaluate surgical resection options.'
    },
    'kidney cancer': {
        'plan': 'Refer to urologist and oncologist; order renal ultrasound, CT/MRI abdomen, and biopsy',
        'lab_tests': [
            {'test': 'CBC', 'description': 'Assess for anemia or polycythemia'},
            {'test': 'Creatinine', 'description': 'Evaluate kidney function'},
            {'test': 'VHL gene testing', 'description': 'Assess for von Hippel-Lindau syndrome'},
            {'test': 'Urinalysis', 'description': 'Monitor for hematuria'}
        ],
        'cancer_follow_up': 'Consider PET/CT for metastasis; monitor for lung or bone involvement; evaluate R.E.N.A.L. nephrometry score.'
    },
    'bladder cancer': {
        'plan': 'Refer to urologist and oncologist; order cystoscopy, biopsy, and CT urography',
        'lab_tests': [
            {'test': 'Urine cytology', 'description': 'Assess for malignant cells in urine'},
            {'test': 'NMP22', 'description': 'Monitor bladder cancer marker'},
            {'test': 'FISH analysis', 'description': 'Detect chromosomal abnormalities'},
            {'test': 'CBC', 'description': 'Evaluate for anemia'}
        ],
        'cancer_follow_up': 'Consider TURBT for staging; monitor for recurrence; evaluate TNM staging.'
    },
    'thyroid cancer': {
        'plan': 'Refer to endocrinologist and oncologist; order thyroid ultrasound, FNA biopsy, and neck CT',
        'lab_tests': [
            {'test': 'TSH', 'description': 'Assess thyroid function'},
            {'test': 'Calcitonin', 'description': 'Monitor for medullary thyroid cancer'},
            {'test': 'RET mutation testing', 'description': 'Evaluate genetic predisposition'},
            {'test': 'Thyroglobulin', 'description': 'Monitor for recurrence in differentiated thyroid cancer'}
        ],
        'cancer_follow_up': 'Consider radioiodine scan; monitor for lymph node metastasis; evaluate AJCC staging.'
    },
    'skin cancer': {
        'plan': 'Refer to dermatologist and oncologist; order skin biopsy and dermoscopy',
        'lab_tests': [
            {'test': 'BRAF mutation testing', 'description': 'Assess for targeted therapy in melanoma'},
            {'test': 'LDH', 'description': 'Evaluate prognosis in advanced melanoma'},
            {'test': 'CBC', 'description': 'Monitor for systemic effects'},
            {'test': 'Skin immunohistochemistry', 'description': 'Confirm melanoma subtype'}
        ],
        'cancer_follow_up': 'Consider sentinel lymph node biopsy for melanoma; monitor for metastasis; evaluate Breslow depth.'
    },
    'brain cancer': {
        'plan': 'Refer to neurologist and neuro-oncologist; order brain MRI, biopsy, and EEG if seizures present',
        'lab_tests': [
            {'test': 'IDH1/2 mutation testing', 'description': 'Assess for glioma prognosis'},
            {'test': 'MGMT methylation', 'description': 'Evaluate for treatment response in glioblastoma'},
            {'test': 'CSF analysis', 'description': 'Assess for leptomeningeal spread'},
            {'test': '1p/19q deletion testing', 'description': 'Evaluate for oligodendroglioma'}
        ],
        'cancer_follow_up': 'Consider functional MRI or PET scan; monitor for tumor progression; evaluate WHO grading.'
    },
    'multiple myeloma': {
        'plan': 'Refer to hematologist/oncologist; order bone marrow biopsy and skeletal survey',
        'lab_tests': [
            {'test': 'Serum protein electrophoresis', 'description': 'Detect monoclonal spike (M protein)'},
            {'test': 'Free light chain assay', 'description': 'Monitor light chain levels'},
            {'test': 'Beta-2 microglobulin', 'description': 'Assess disease burden'},
            {'test': 'CRAB criteria labs', 'description': 'Monitor calcium, renal function, anemia, bone lesions'}
        ],
        'cancer_follow_up': 'Consider PET/CT for lytic lesions; monitor for renal failure; evaluate ISS staging.'
    },
    'sarcoma': {
        'plan': 'Refer to orthopedic oncologist and oncologist; order MRI of mass, biopsy, and chest CT',
        'lab_tests': [
            {'test': 'LDH', 'description': 'Assess tumor activity'},
            {'test': 'Cytogenetic testing', 'description': 'Identify sarcoma-specific translocations (e.g., EWS-FLI1 in Ewing sarcoma)'},
            {'test': 'CBC', 'description': 'Monitor for anemia or bone marrow involvement'},
            {'test': 'ALP', 'description': 'Evaluate bone involvement in osteosarcoma'}
        ],
        'cancer_follow_up': 'Consider PET/CT for metastasis; monitor for local recurrence; evaluate surgical margins.'
    },
    'testicular cancer': {
        'plan': 'Refer to urologist and oncologist; order scrotal ultrasound, biopsy, and chest/abdominal CT',
        'lab_tests': [
            {'test': 'Beta-hCG', 'description': 'Monitor tumor marker for germ cell tumors'},
            {'test': 'AFP', 'description': 'Monitor for non-seminomatous tumors'},
            {'test': 'LDH', 'description': 'Assess tumor burden'},
            {'test': 'Testosterone', 'description': 'Evaluate testicular function'}
        ],
        'cancer_follow_up': 'Consider retroperitoneal lymph node dissection; monitor for recurrence; evaluate IGCCCG risk classification.'
    },
    'endometrial cancer': {
        'plan': 'Refer to gynecologic oncologist; order endometrial biopsy, transvaginal ultrasound, and pelvic MRI',
        'lab_tests': [
            {'test': 'CA-125', 'description': 'Monitor tumor marker for advanced disease'},
            {'test': 'CBC', 'description': 'Assess for anemia due to bleeding'},
            {'test': 'MSI/MMR testing', 'description': 'Evaluate for Lynch syndrome'},
            {'test': 'ER/PR status', 'description': 'Determine hormone receptor status'}
        ],
        'cancer_follow_up': 'Consider PET/CT for staging; monitor for lymph node involvement; evaluate FIGO staging.'
    },
    'gallbladder cancer': {
        'plan': 'Refer to hepatobiliary surgeon and oncologist; order abdominal ultrasound, CT/MRI, and biopsy',
        'lab_tests': [
            {'test': 'CEA', 'description': 'Monitor tumor marker'},
            {'test': 'CA 19-9', 'description': 'Assess gallbladder cancer activity'},
            {'test': 'LFTs', 'description': 'Evaluate liver function and jaundice'},
            {'test': 'CBC', 'description': 'Monitor for anemia'}
        ],
        'cancer_follow_up': 'Consider ERCP or MRCP for biliary involvement; monitor for liver metastasis; evaluate TNM staging.'
    },
    'mesothelioma': {
        'plan': 'Refer to thoracic oncologist; order chest CT, pleural biopsy, and thoracoscopy',
        'lab_tests': [
            {'test': 'Mesothelin', 'description': 'Monitor mesothelioma marker'},
            {'test': 'CBC', 'description': 'Assess for anemia or inflammation'},
            {'test': 'LDH', 'description': 'Evaluate disease activity'},
            {'test': 'Pleural fluid analysis', 'description': 'Assess for malignant cells'}
        ],
        'cancer_follow_up': 'Consider PET/CT for staging; monitor for pleural effusion; evaluate IMIG staging.'
    },
    'oral cancer': {
        'plan': 'Refer to head and neck oncologist; order oral biopsy, neck CT/MRI, and panendoscopy',
        'lab_tests': [
            {'test': 'HPV testing', 'description': 'Assess for HPV-related oropharyngeal cancer'},
            {'test': 'CBC', 'description': 'Monitor for anemia or infection'},
            {'test': 'LFTs', 'description': 'Evaluate for metastasis'},
            {'test': 'SCC antigen', 'description': 'Monitor squamous cell carcinoma activity'}
        ],
        'cancer_follow_up': 'Consider PET/CT for staging; monitor for lymph node metastasis; evaluate AJCC staging.'
    }
}


# ~/projects/hospital/departments/nlp/resources/cancer_diseases.py
import os

CANCER_KEYWORDS_FILE = os.path.join(os.path.dirname(__file__), "cancer_keywords.json")