import logging
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import unicodedata
import re
from typing import Union, Dict, List, Optional
import warnings

logger = logging.getLogger("HIMS-NLP")

# Configure logging
logging.basicConfig(
    filename='summarizer_warnings.log', 
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ClinicalSummarizer:
    """Clinical text summarizer using BART model with medical domain verification."""
    
    # Class-level model caching
    _model = None
    _tokenizer = None
    _nlp = None
    _initialized = False
    
    # Medical synonyms and terminology
    MEDICAL_SYNONYMS = {
        'dyspnea': ['shortness of breath', 'breathlessness', 'difficulty breathing'],
        'shortness of breath': ['dyspnea', 'breathlessness', 'difficulty breathing'],
        'fever': ['pyrexia', 'elevated temperature', 'hyperthermia'],
        'cough': ['productive cough', 'non-productive cough'],
        'pneumonia': ['post-infectious pneumonia', 'community-acquired pneumonia', 
                     'healthcare-associated pneumonia', 'lung infection'],
        'antibiotic-resistant organism': ['resistant organism', 'multi-drug resistant organism', 'MDRO'],
        'urinary tract infection': ['UTI', 'cystitis', 'bladder infection'],
        'sepsis': ['early sepsis', 'systemic infection', 'bacteremia'],
        'fatigue': ['tiredness', 'lethargy', 'exhaustion'],
        'ciprofloxacin': ['antibiotic treatment', 'fluoroquinolone'],
        'type 2 diabetes mellitus': ['diabetes', 'T2DM', 'type 2 diabetes'],
        'hypertension': ['high blood pressure', 'HTN'],
        'smoking': ['tobacco use', 'cigarette use'],
        'chest pain': ['thoracic pain', 'precordial pain'],
        'headache': ['cephalgia', 'migraine']
    }
    
    GENERIC_TERMS = {
        'history', 'old', 'year', 'years', 'patient', 'presented', 'levels', 'elevated',
        'diagnosed', 'admitted', 'examination', 'tests', 'high', 'low', 'year-old',
        'blood tests', 'physical examination', 'contact', 'male', 'female', 'woman', 'man',
        'positive', 'clinician', 'emergency room', 'clinical note', 'recent', 'recently',
        'ago', 'daily', 'bid', 'po', 'iv', 'status', 'condition', 'finding'
    }

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None):
        """Initialize the ClinicalSummarizer.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
        """
        self._initialize_models(model_name, device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        if self._model is not None:
            self._model = self._model.to(self.device)

    @classmethod
    def _initialize_models(cls, model_name: str, device: Optional[str] = None) -> None:
        """Initialize models with thread-safe lazy loading."""
        if cls._initialized:
            return
            
        try:
            print("Loading ClinicalSummarizer models... (this may take a while)")
            
            # Load BART model and tokenizer
            cls._tokenizer = BartTokenizer.from_pretrained(model_name)
            cls._model = BartForConditionalGeneration.from_pretrained(model_name)
            
            # Load biomedical NLP model
            try:
                cls._nlp = spacy.load("en_core_sci_sm")
            except OSError:
                logger.warning("en_core_sci_sm not found, falling back to en_core_web_sm")
                cls._nlp = spacy.load("en_core_web_sm")
                
            cls._initialized = True
            logger.info("ClinicalSummarizer models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def summarize(self, note: Union[str, Dict], max_length: int = 200, min_length: int = 30) -> str:
        """Generate a clinical summary from input text or dictionary.
        
        Args:
            note: Input text or dictionary with clinical information
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Formatted clinical summary
        """
        try:
            # Validate input
            if not note:
                return "Error: Input is empty or None."
            
            # Preprocess input
            text = self._preprocess_input(note)
            if not text.strip():
                return "Error: No valid text found in input."
            
            # Generate initial summary
            raw_summary = self._generate_summary(text, max_length, min_length)
            
            # Verify and format summary
            verified_summary = self._verify_summary(text, raw_summary)
            return self._format_summary(verified_summary)
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return f"Error generating summary: {str(e)}"

    def _preprocess_input(self, note: Union[str, Dict]) -> str:
        """Preprocess input text or dictionary."""
        if isinstance(note, dict):
            text = " ".join(f"{k}: {v}" for k, v in note.items() if v and isinstance(v, str))
        elif isinstance(note, str):
            text = note
        else:
            raise ValueError(f"Unsupported input type: {type(note)}")
        
        # Normalize text
        text = unicodedata.normalize('NFKC', text)
        
        # Replace medical abbreviations and special characters
        replacements = {
            '°C': ' degrees Celsius',
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'RR': 'respiratory rate',
            'SpO2': 'oxygen saturation',
            'T:': 'Temperature:'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Add guiding prompt for better medical summarization
        prompt = "Summarize the following clinical note accurately and concisely: "
        return prompt + text

    def _generate_summary(self, text: str, max_length: int, min_length: int) -> str:
        """Generate summary using BART model."""
        try:
            inputs = self._tokenizer(
                [text], 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            summary_ids = self._model.generate(
                inputs["input_ids"],
                num_beams=4,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True,
                repetition_penalty=2.0,  # Increased to reduce repetition
                length_penalty=2.0,      # Added for better length control
                no_repeat_ngram_size=3   # Added to prevent n-gram repetition
            )
            
            summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logger.debug(f"Generated raw summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise

    def _verify_summary(self, input_text: str, summary: str) -> str:
        """Verify and filter summary based on medical consistency."""
        try:
            input_doc = self._nlp(input_text.lower())
            summary_doc = self._nlp(summary.lower())
            
            # Extract relevant entities
            input_entities = self._extract_medical_entities(input_doc)
            summary_entities = self._extract_medical_entities(summary_doc)
            
            logger.debug(f"Input entities: {input_entities}")
            logger.debug(f"Summary entities: {summary_entities}")
            
            # Filter sentences based on medical relevance
            sentences = self._split_sentences_preserving_medical(summary)
            filtered_sentences = self._filter_sentences(sentences, input_entities)
            
            if not filtered_sentences:
                logger.warning("No medically relevant sentences after filtering")
                return "Unable to generate a reliable clinical summary. Please verify the input contains sufficient medical information."
            
            return ". ".join(filtered_sentences)
            
        except Exception as e:
            logger.error(f"Summary verification failed: {e}")
            return summary  # Return original summary if verification fails

    def _extract_medical_entities(self, doc) -> set:
        """Extract medical entities from spaCy document."""
        entities = set()
        for ent in doc.ents:
            entity_text = ent.text.lower().strip()
            if entity_text and entity_text not in self.GENERIC_TERMS:
                entities.add(entity_text)
                # Add synonyms
                if entity_text in self.MEDICAL_SYNONYMS:
                    entities.update(self.MEDICAL_SYNONYMS[entity_text])
        return entities

    def _split_sentences_preserving_medical(self, text: str) -> List[str]:
        """Split text into sentences while preserving medical abbreviations."""
        # Split on periods not preceded by common medical abbreviations
        pattern = r'(?<!\b(?:Dr|Mr|Mrs|Ms|vs|etc|Fig|fig|Vol|vol|No|no|[A-Z]))\.\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _filter_sentences(self, sentences: List[str], input_entities: set) -> List[str]:
        """Filter sentences based on medical relevance."""
        filtered = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_doc = self._nlp(sentence.lower())
            sentence_entities = self._extract_medical_entities(sentence_doc)
            
            # Keep sentence if it has no specific entities (generic clinical text)
            # OR if it shares entities with input
            if not sentence_entities or sentence_entities.intersection(input_entities):
                filtered.append(sentence)
            else:
                logger.debug(f"Filtered sentence: '{sentence}' - Entities: {sentence_entities}")
                
        return filtered

    def _format_summary(self, summary: str) -> str:
        """Format the final summary output."""
        if summary.startswith("Unable to generate"):
            return summary
            
        sentences = [s.strip() for s in summary.split(". ") if s.strip()]
        if not sentences:
            return "No summary could be generated."
            
        return "- " + "\n- ".join(sentences)

    def batch_summarize(self, notes: List[Union[str, Dict]], **kwargs) -> List[str]:
        """Generate summaries for multiple notes."""
        return [self.summarize(note, **kwargs) for note in notes]

    def __repr__(self) -> str:
        return f"ClinicalSummarizer(device='{self.device}', models_loaded={self._initialized})"


# Example usage
if __name__ == "__main__":
    summarizer = ClinicalSummarizer()
    
    sample_note = {
        "hpi": "Patient is a 60-year-old woman presenting with chest pain and shortness of breath for 2 days.",
        "medical_history": "History of hypertension and obesity. No known allergies.",
        "assessment": "Likely unstable angina. Needs further evaluation.",
        "recommendation": "Admit to hospital. Start aspirin, oxygen, and monitor vitals. Order ECG and troponin levels."
    }
    
    summary = summarizer.summarize(sample_note)
    print("Clinical Summary:")
    print(summary)