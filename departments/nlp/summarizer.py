import logging
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import unicodedata
import re

logger = logging.getLogger("HIMS-NLP")

# Configure logging to a file for technical details
logging.basicConfig(filename='summarizer_warnings.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ClinicalSummarizer:
    _model = None
    _tokenizer = None
    _nlp = None

    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        if ClinicalSummarizer._model is None or ClinicalSummarizer._tokenizer is None:
            print("Loading model... (this may take a while the first time)")
            ClinicalSummarizer._tokenizer = BartTokenizer.from_pretrained(model_name)
            ClinicalSummarizer._model = BartForConditionalGeneration.from_pretrained(model_name)
            try:
                ClinicalSummarizer._nlp = spacy.load("en_core_sci_sm")  # Load biomedical NLP model
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {e}")
                raise
        
        self.model = ClinicalSummarizer._model
        self.tokenizer = ClinicalSummarizer._tokenizer
        self.nlp = ClinicalSummarizer._nlp
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def summarize(self, note, max_length=200, min_length=30):
        if isinstance(note, dict):
            text = " ".join(f"{k}: {v}" for k, v in note.items() if v)
        elif isinstance(note, str):
            text = note
        else:
            return "Error: Input must be a text string or dictionary."
        
        # Normalize special characters
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('°C', ' degrees Celsius')
        
        # Add a guiding prompt
        prompt = "Summarize the following clinical note accurately, including only information explicitly stated in a concise manner: "
        text = prompt + text
        
        inputs = self.tokenizer([text], max_length=512, truncation=True, return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            repetition_penalty=1.5
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.debug(f"Raw summary: {summary}")
        verified_summary = self.verify_summary(text, summary)
        return self.format_summary(verified_summary)

    def verify_summary(self, input_text, summary):
        # Define synonym mappings
        synonyms = {
            'dyspnea': ['shortness of breath', 'breathlessness'],
            'shortness of breath': ['dyspnea', 'breathlessness'],
            'fever': ['pyrexia', 'elevated temperature'],
            'cough': ['productive cough'],
            'pneumonia': ['post-infectious pneumonia', 'community-acquired pneumonia', 'healthcare-associated pneumonia'],
            'antibiotic-resistant organism': ['resistant organism', 'multi-drug resistant organism'],
            'urinary tract infection': ['UTI', 'cystitis'],
            'sepsis': ['early sepsis', 'systemic infection'],
            'fatigue': ['tiredness', 'lethargy'],
            'ciprofloxacin': ['antibiotic treatment'],
            'type 2 diabetes mellitus': ['diabetes', 'T2DM'],
            'hypertension': ['high blood pressure'],
            'smoking': ['tobacco use']
        }

        # Process input and summary with spaCy
        input_doc = self.nlp(input_text)
        summary_doc = self.nlp(summary)
        
        # Extract clinically relevant entities and filter out generic terms
        generic_terms = {
            'history', 'old', 'year', 'years', 'patient', 'presented', 'levels', 'elevated',
            'diagnosed', 'admitted', 'examination', 'tests', 'high', 'low', '60-year', '45-year',
            'blood tests', 'physical examination', 'contact', 'male', 'female', 'woman', 'man',
            'positive', 'clinician', 'emergency room', 'clinical note', 'recent', 'recently',
            'ago', 'daily', 'bid', 'po', 'iv'
        }
        input_entities = set(ent.text.lower() for ent in input_doc.ents if ent.text.lower() not in generic_terms)
        summary_entities = set(ent.text.lower() for ent in summary_doc.ents if ent.text.lower() not in generic_terms)
        
        logger.debug(f"Input entities: {input_entities}")
        logger.debug(f"Summary entities: {summary_entities}")
        
        # Adjust for synonyms
        adjusted_input_entities = set(input_entities)
        for ent in input_entities:
            if ent in synonyms:
                adjusted_input_entities.update(synonyms[ent])
        
        # Split summary into sentences, preserving medical abbreviations
        sentences = re.split(r'(?<!\d|[A-Z])\. (?!\d|[A-Z])', summary)
        logger.debug(f"Split sentences: {sentences}")
        
        filtered_sentences = []
        for s in sentences:
            if not s.strip():
                continue
            sentence_doc = self.nlp(s)
            sentence_entities = set(ent.text.lower() for ent in sentence_doc.ents if ent.text.lower() not in generic_terms)
            # Include sentence if it contains at least one valid entity or no entities
            if not sentence_entities or any(se in adjusted_input_entities for se in sentence_entities):
                filtered_sentences.append(s.strip())
            else:
                logger.debug(f"Filtering sentence '{s}' due to unsupported entities: {sentence_entities - adjusted_input_entities}")
        
        # If no valid sentences remain, return a fallback
        if not filtered_sentences:
            logger.warning("No valid sentences after filtering. Returning fallback.")
            return "Unable to generate a reliable summary. Please check the input note."
        
        summary = ". ".join(filtered_sentences)
        return summary

    def format_summary(self, summary):
        if summary.startswith("Unable to generate"):
            return summary
        sentences = [s.strip() for s in summary.split(". ") if s.strip()]
        formatted = "- " + "\n- ".join(sentences)
        return formatted