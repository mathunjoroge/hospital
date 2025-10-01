import logging
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import unicodedata
import json
from typing import Union, Dict, List, Optional, Set
import warnings

# --- Basic Configuration ---

logger = logging.getLogger("HIMS-NLP")

# Configure logging to file
logging.basicConfig(
    filename='summarizer_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Clinical Summarizer Class ---

class ClinicalSummarizer:
    """
    An advanced clinical text summarizer using a BART model for summarization
    and an enhanced AI simulation for providing detailed clinical suggestions.
    """

    # Class-level cache for models to avoid reloading
    _model: Optional[BartForConditionalGeneration] = None
    _tokenizer: Optional[BartTokenizer] = None
    _nlp: Optional[spacy.Language] = None
    _initialized: bool = False

    GENERIC_TERMS: Set[str] = {
        'history', 'old', 'year', 'years', 'patient', 'presented', 'levels', 'elevated',
        'diagnosed', 'admitted', 'examination', 'tests', 'high', 'low', 'year-old',
        'blood tests', 'physical examination', 'contact', 'male', 'female', 'woman', 'man',
        'positive', 'clinician', 'emergency room', 'clinical note', 'recent', 'recently',
        'ago', 'daily', 'bid', 'po', 'iv', 'status', 'condition', 'finding'
    }

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models(model_name)

        if self._model:
            self._model.to(self.device)

    @classmethod
    def _initialize_models(cls, model_name: str) -> None:
        if cls._initialized:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                print("Loading ClinicalSummarizer models... (this may take a while on first run)")

                cls._tokenizer = BartTokenizer.from_pretrained(model_name)
                cls._model = BartForConditionalGeneration.from_pretrained(model_name)

                try:
                    cls._nlp = spacy.load("en_core_sci_sm")
                except OSError:
                    logger.warning("en_core_sci_sm not found. Falling back to en_core_web_sm.")
                    cls._nlp = spacy.load("en_core_web_sm")

                cls._initialized = True
                logger.info(f"ClinicalSummarizer models loaded successfully onto device: {cls._model.device if cls._model else 'CPU'}")

            except Exception as e:
                logger.error(f"Failed to initialize models: {e}")
                raise RuntimeError(f"Model initialization failed: {e}")

    def summarize(self, note: Union[str, Dict], max_length: int = 200, min_length: int = 40) -> str:
        try:
            if not note:
                return "<p><strong>Error:</strong> Input note is empty or None.</p>"

            text = self._preprocess_input(note)
            if not text.strip():
                return "<p><strong>Error:</strong> No valid text content found in the input.</p>"

            raw_summary = self._generate_summary(text, max_length, min_length)
            verified_summary = self._verify_summary(text, raw_summary)

            if not verified_summary or "Unable to generate" in verified_summary:
                return f"<p>{verified_summary}</p>"

            suggestions = self._generate_suggestions(verified_summary)
            return self._format_output(verified_summary, suggestions)

        except Exception as e:
            logger.error(f"An unexpected error occurred during summarization: {e}", exc_info=True)
            return f"<p><strong>Error:</strong> An unexpected error occurred. Check logs for details.</p>"

    def _preprocess_input(self, note: Union[str, Dict]) -> str:
        if isinstance(note, dict):
            text = ". ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in note.items() if v and isinstance(v, str))
        elif isinstance(note, str):
            text = note
        else:
            raise TypeError(f"Unsupported input type: {type(note)}. Must be str or dict.")

        text = unicodedata.normalize('NFKC', text)
        replacements = {'°C': ' degrees Celsius', 'BP': 'blood pressure', 'HR': 'heart rate'}
        for old, new in replacements.items():
            text = text.replace(old, new)

        return "Summarize the following clinical note accurately and concisely: " + text

    def _generate_summary(self, text: str, max_length: int, min_length: int) -> str:
        try:
            inputs = self._tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(self.device)
            summary_ids = self._model.generate(
                inputs["input_ids"], num_beams=4, max_length=max_length, min_length=min_length, early_stopping=True
            )
            summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logger.debug(f"Generated raw summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Core summary generation failed: {e}", exc_info=True)
            raise

    def _verify_summary(self, input_text: str, summary: str) -> str:
        try:
            input_doc = self._nlp(input_text.lower())
            input_entities = self._extract_medical_entities(input_doc)
            sentences = self._split_sentences(summary)
            filtered_sentences = self._filter_sentences(sentences, input_entities)

            if not filtered_sentences:
                logger.warning("No medically relevant sentences found after filtering.")
                return "Unable to generate a reliable clinical summary due to lack of specific medical information."
            
            return " ".join(filtered_sentences).strip()
        except Exception as e:
            logger.error(f"Summary verification step failed: {e}", exc_info=True)
            return summary

    def _extract_medical_entities(self, doc: spacy.tokens.Doc) -> Set[str]:
        entities = set()
        for ent in doc.ents:
            entity_text = ent.text.lower().strip()
            if entity_text and entity_text not in self.GENERIC_TERMS:
                entities.add(entity_text)
        return entities

    def _split_sentences(self, text: str) -> List[str]:
        if not self._nlp:
            return [s.strip() for s in text.split('.') if s]
        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _filter_sentences(self, sentences: List[str], input_entities: Set[str]) -> List[str]:
        filtered = []
        for sentence in sentences:
            sentence_doc = self._nlp(sentence.lower())
            sentence_entities = self._extract_medical_entities(sentence_doc)
            if not sentence_entities or sentence_entities.intersection(input_entities):
                filtered.append(sentence)
        return filtered

    def _generate_suggestions(self, summary_text: str) -> Dict[str, List[str]]:
        """Generates clinical suggestions using a simulated, context-aware AI."""
        # This method orchestrates the creation of a prompt and the parsing of the
        # response from our more advanced simulation engine.
        prompt = self._create_suggestion_prompt(summary_text)
        logger.info("Generating comprehensive, context-aware suggestions based on summary.")
        simulated_response = self._simulate_comprehensive_llm_call(summary_text)
        return self._parse_llm_response(simulated_response)

    def _create_suggestion_prompt(self, summary_text: str) -> str:
        """Creates a structured prompt for the generative AI."""
        return f"""
        Analyze the following clinical summary as a clinical decision support tool. 
        Based ONLY on the information provided, provide:
        1. A list of potential differential diagnoses, ordered by likelihood.
        2. A list of key management and work-up suggestions.

        Clinical Summary: "{summary_text}"

        Respond in a valid JSON format with keys "potential_diagnoses" and "management_suggestions".
        """

    # --- NEW: This simulation is now more comprehensive and clinically structured ---
    def _simulate_comprehensive_llm_call(self, summary_text: str) -> str:
        """
        Simulates a more advanced LLM call by performing a structured analysis
        of the summary text to generate a detailed and relevant set of suggestions.
        """
        diagnoses = []
        management = []
        lower_summary = summary_text.lower()

        # --- Clinical Picture Flags ---
        has_respiratory_symptoms = all(k in lower_summary for k in ['fever', 'cough'])
        has_recent_antibiotics = 'ciprofloxacin' in lower_summary or 'recent antibiotic' in lower_summary
        
        # --- Diagnosis Generation ---
        if has_respiratory_symptoms:
            primary_dx = "Community-Acquired Pneumonia (CAP)"
            if has_recent_antibiotics:
                # Add nuance if there's a history of antibiotic use
                diagnoses.append(f"{primary_dx}, with increased risk for antibiotic-resistant organism.")
                diagnoses.append("Healthcare-Associated Pneumonia (HCAP) if recent healthcare contact.")
            else:
                diagnoses.append(primary_dx)
            
            diagnoses.append("Acute Bronchitis (viral or bacterial).")
            diagnoses.append("Influenza-like illness.")

        # --- Management Plan Generation ---
        if has_respiratory_symptoms:
            management.append("Order a Chest X-ray (PA and Lateral) to assess for consolidation.")
            management.append("Obtain baseline labs: CBC with differential, C-reactive protein (CRP), and basic metabolic panel (BMP).")
            management.append("Obtain blood and sputum cultures *before* starting antibiotics to guide therapy.")
            management.append("Monitor vital signs and oxygen saturation (SpO2) closely.")
            
            if has_recent_antibiotics:
                management.append("Initiate broad-spectrum empirical antibiotics covering resistant pathogens, guided by local antibiogram (e.g., Vancomycin + Piperacillin-Tazobactam).")
            else:
                management.append("Start empirical antibiotics for typical CAP (e.g., Ceftriaxone plus Azithromycin).")

        # --- Default suggestions if no specific pattern is matched ---
        if not diagnoses:
            diagnoses.append("Diagnosis unclear from summary; requires further clinical correlation.")
        if not management:
            management.append("Standard clinical work-up based on the patient's full history and examination is advised.")

        response_data = {
            "potential_diagnoses": diagnoses,
            "management_suggestions": management
        }
        
        return json.dumps(response_data)

    def _parse_llm_response(self, response_json: str) -> Dict[str, List[str]]:
        """Parses and validates the JSON response from the LLM simulation."""
        try:
            data = json.loads(response_json)
            return {
                "diagnosis": data.get("potential_diagnoses", []),
                "management": data.get("management_suggestions", [])
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return {"diagnosis": [], "management": []}

    def _format_output(self, summary: str, suggestions: Dict[str, List[str]]) -> str:
        output_parts = ["<h2>Clinical Summary</h2>"]
        sentences = self._split_sentences(summary)
        
        if sentences:
            output_parts.append("<ul>\n" + "\n".join(f"<li>{s.rstrip('.')}.</li>" for s in sentences) + "\n</ul>")
        else:
            output_parts.append("<p>No summary could be generated.</p>")

        if suggestions.get('diagnosis'):
            output_parts.append("<h3>Potential Diagnoses (AI-Generated)</h3>")
            output_parts.append("<ul>\n<li>" + "</li>\n<li>".join(suggestions['diagnosis']) + "</li>\n</ul>")

        if suggestions.get('management'):
            output_parts.append("<h3>Management Suggestions (AI-Generated)</h3>")
            output_parts.append("<ul>\n<li>" + "</li>\n<li>".join(suggestions['management']) + "</li>\n</ul>")
        
        if suggestions.get('diagnosis') or suggestions.get('management'):
            output_parts.append("<hr>")
            output_parts.append(
                "<p><strong>Disclaimer:</strong> These AI-generated suggestions are for "
                "informational purposes only and are not a substitute for professional "
                "clinical judgment. All diagnostic and treatment decisions must be made "
                "by a qualified healthcare provider.</p>"
            )

        return "\n".join(output_parts)

    def batch_summarize(self, notes: List[Union[str, Dict]], **kwargs) -> List[str]:
        return [self.summarize(note, **kwargs) for note in notes]

    def __repr__(self) -> str:
        return f"ClinicalSummarizer(device='{self.device}', models_loaded={self._initialized})"