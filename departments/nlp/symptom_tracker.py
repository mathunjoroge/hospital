# departments/nlp/symptom_tracker.py

class SymptomTracker:
    def __init__(self):
        self.common_symptoms = {
            'neurological': {
                'headache': 'Pain or discomfort in the head or scalp.',
                'photophobia': 'Sensitivity to light.',
                'dizziness': 'Feeling of spinning or unsteadiness.',
                'numbness': 'Loss of sensation in a body part.',
                'tingling': 'Pins and needles sensation.',
                'tremors': 'Involuntary shaking movements.',
                'seizures': 'Sudden, uncontrolled electrical disturbance in the brain.',
                'memory loss': 'Difficulty recalling information.',
                'confusion': 'Disorientation or unclear thinking.',
                'loss of balance': 'Difficulty maintaining physical stability.',
                'difficulty concentrating': 'Trouble focusing on tasks.',
                'speech difficulties': 'Problems articulating words.',
                'vision changes': 'Alterations in visual perception.',
                'double vision': 'Seeing two images of a single object.',
                'blurred vision': 'Lack of sharpness in vision.'
            },
            'cardiovascular': {
                'chest pain': 'Discomfort or pain in the chest area.',
                'shortness of breath': 'Difficulty breathing or feeling of suffocation.',
                'palpitations': 'Sensation of rapid or irregular heartbeat.',
                'lightheadedness': 'Feeling faint or about to pass out.',
                'sweating': 'Excessive perspiration, often cold or clammy.',
                'cyanosis': 'Bluish discoloration of skin due to low oxygen.',
                'pallor': 'Unusual paleness of skin.',
                'irregular heartbeat': 'Abnormal heart rhythm.',
                'high blood pressure': 'Elevated blood pressure readings.',
                'low blood pressure': 'Abnormally low blood pressure.',
                'fainting': 'Temporary loss of consciousness.',
                'shock': 'Life-threatening condition due to inadequate blood flow.'
            },
            'gastrointestinal': {
                'nausea': 'Feeling of needing to vomit.',
                'vomiting': 'Expulsion of stomach contents through the mouth.',
                'epigastric pain': 'Pain in the upper central abdomen.',
                'diarrhea': 'Frequent, loose, or watery stools.',
                'constipation': 'Infrequent or difficult bowel movements.',
                'abdominal discomfort': 'General unease or pain in the abdomen.',
                'abdominal bloating': 'Feeling of fullness or swelling in the abdomen.',
                'gas': 'Excessive air in the digestive tract.',
                'heartburn': 'Burning sensation in the chest due to acid reflux.',
                'hiccups': 'Involuntary contractions of the diaphragm.',
                'belching': 'Expelling air from the stomach through the mouth.',
                'dark stools': 'Stools that are black or tarry, possibly indicating bleeding.',
                'difficulty swallowing': 'Trouble passing food or liquid through the throat.'
            },
            'respiratory': {
                'cough': 'Sudden expulsion of air from the lungs.',
                'wheezing': 'High-pitched sound during breathing.',
                'difficulty breathing': 'Labored or restricted breathing.',
                'chest tightness': 'Feeling of pressure or constriction in the chest.',
                'nasal congestion': 'Blocked or stuffy nose.',
                'runny nose': 'Excessive nasal discharge.',
                'sinus pressure': 'Pain or pressure around the sinuses.'
            },
            'musculoskeletal': {
                'knee pain': 'Discomfort or pain in the knee joint.',
                'back pain': 'Pain in the lower, middle, or upper back.',
                'joint pain': 'Discomfort in one or more joints.',
                'muscle weakness': 'Reduced strength in muscles.',
                'muscle cramps': 'Sudden, involuntary muscle contractions.',
                'muscle spasms': 'Prolonged involuntary muscle contractions.',
                'joint stiffness': 'Reduced ease of movement in joints.',
                'joint swelling': 'Enlargement of joints due to fluid or inflammation.',
                'shoulder pain': 'Discomfort in the shoulder area.',
                'hip pain': 'Pain in the hip joint or surrounding area.',
                'foot pain': 'Discomfort in the foot.',
                'heel pain': 'Pain in the heel area.',
                'calf pain': 'Discomfort in the calf muscle.',
                'difficulty moving': 'Trouble initiating or completing movements.',
                'joint locking': 'Joint getting stuck in one position.',
                'joint instability': 'Feeling that a joint is loose or unsteady.'
            },
            'dermatological': {
                'rash': 'Change in skin color or texture, often red or bumpy.',
                'itching': 'Uncomfortable sensation prompting scratching.',
                'skin discoloration': 'Changes in skin tone or pigmentation.',
                'bruising': 'Discoloration due to blood under the skin.',
                'skin lesions': 'Abnormal growths or patches on the skin.',
                'skin ulcers': 'Open sores on the skin.',
                'skin thickening': 'Increased skin density or hardness.',
                'skin peeling': 'Flaking or shedding of the outer skin layer.',
                'redness': 'Skin appearing red or flushed.'
            },
            'general': {
                'fatigue': 'Persistent tiredness or lack of energy.',
                'fever': 'Elevated body temperature above normal.',
                'chills': 'Feeling cold with shivering, often with fever.',
                'edema': 'Swelling due to fluid accumulation.',
                'weight loss': 'Unintentional reduction in body weight.',
                'weight gain': 'Unintentional increase in body weight.',
                'night sweats': 'Excessive sweating during sleep.',
                'frequent infections': 'Recurrent or persistent infections.',
                'slow wound healing': 'Delayed recovery of skin or tissue injuries.'
            },
            'other': {
                'sore throat': 'Pain or irritation in the throat.',
                'ear pain': 'Discomfort or pain in the ear.',
                'loss of appetite': 'Reduced desire to eat.',
                'insomnia': 'Difficulty falling or staying asleep.',
                'hoarseness': 'Raspy or strained voice.',
                'burning sensation': 'Feeling of heat or stinging in a body part.',
                'urinary frequency': 'Need to urinate more often than usual.',
                'urinary urgency': 'Sudden, compelling need to urinate.',
                'blood in urine': 'Presence of blood in the urine.',
                'cold intolerance': 'Increased sensitivity to cold temperatures.',
                'heat intolerance': 'Increased sensitivity to warm temperatures.',
                'hair loss': 'Unusual loss of hair from the scalp or body.',
                'hair thinning': 'Reduction in hair density.',
                'nail changes': 'Abnormalities in nail color, shape, or texture.',
                'bleeding': 'Unusual or excessive blood loss.',
                'anxiety': 'Feelings of worry or unease.',
                'depression': 'Persistent sadness or loss of interest.',
                'irritability': 'Increased tendency to become annoyed.',
                'restlessness': 'Inability to stay calm or still.',
                'dry mouth': 'Lack of saliva causing a dry feeling in the mouth.',
                'mouth sores': 'Painful lesions in the mouth.',
                'tooth pain': 'Discomfort or pain in the teeth.',
                'gum bleeding': 'Bleeding from the gums.',
                'neck stiffness': 'Reduced flexibility or pain in the neck.',
                'leg swelling': 'Fluid accumulation in the legs.',
                'ankle swelling': 'Fluid accumulation in the ankles.',
                'weakness': 'General reduction in physical strength.',
                'loss of sensation': 'Inability to feel touch or other sensations.',
                'pins and needles': 'Tingling or prickling sensation.',
                'snoring': 'Noisy breathing during sleep.',
                'daytime sleepiness': 'Excessive tiredness during the day.',
                'loss of consciousness': 'Sudden unresponsiveness.',
                'eye pain': 'Discomfort or pain in the eye.',
                'red eyes': 'Bloodshot or red appearance of the eyes.',
                'dry eyes': 'Insufficient tear production causing dryness.',
                'watery eyes': 'Excessive tear production.',
                'ear ringing': 'Perception of ringing or buzzing in the ears.',
                'hearing loss': 'Reduced ability to hear sounds.',
                'nosebleeds': 'Bleeding from the nasal passages.',
                'difficulty hearing': 'Trouble perceiving sounds clearly.',
                'taste changes': 'Altered perception of taste.',
                'smell changes': 'Altered perception of odors.',
                'abdominal pain': 'Pain in the abdominal region.',
                'pelvic pain': 'Discomfort in the lower abdomen or pelvis.',
                'genital pain': 'Pain in the genital area.',
                'painful urination': 'Discomfort or burning during urination.',
                'incontinence': 'Involuntary loss of urine or feces.',
                'sexual dysfunction': 'Problems with sexual function or performance.',
                'menstrual irregularities': 'Abnormalities in menstrual cycle.',
                'hot flashes': 'Sudden feelings of warmth, often with sweating.',
                'breast pain': 'Discomfort or pain in the breast tissue.',
                'breast lump': 'Palpable mass in the breast.',
                'nipple discharge': 'Fluid leaking from the nipple.',
                'swollen glands': 'Enlarged lymph nodes.',
                'lymph node enlargement': 'Increased size of lymph nodes.',
                'abnormal heart sounds': 'Unusual sounds during heartbeat.',
                'jaundice': 'Yellowing of the skin or eyes.',
                'abdominal mass': 'Palpable lump in the abdomen.'
            }
        }

    def add_symptom(self, category, symptom, description):
        """Add a new symptom to the specified category."""
        if category in self.common_symptoms:
            self.common_symptoms[category][symptom] = description
        else:
            self.common_symptoms[category] = {symptom: description}

    def remove_symptom(self, category, symptom):
        """Remove a symptom from the specified category."""
        if category in self.common_symptoms and symptom in self.common_symptoms[category]:
            del self.common_symptoms[category][symptom]
            if not self.common_symptoms[category]:
                del self.common_symptoms[category]

    def get_symptoms_by_category(self, category):
        """Return all symptoms in a specific category."""
        return self.common_symptoms.get(category, {})

    def search_symptom(self, symptom):
        """Search for a symptom across all categories."""
        for category, symptoms in self.common_symptoms.items():
            if symptom in symptoms:
                return category, symptoms[symptom]
        return None, None

    def get_all_symptoms(self):
        """Return a flat set of all symptom names."""
        return {symptom for category in self.common_symptoms.values() for symptom in category.keys()}

    def get_categories(self):
        """Return a list of all symptom categories."""
        return list(self.common_symptoms.keys())