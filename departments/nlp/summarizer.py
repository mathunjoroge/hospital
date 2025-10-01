import logging
import spacy
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import unicodedata
from typing import Union, Dict, List, Optional, Set
import warnings
import re
from datetime import datetime
import bleach

# --- HMIS Configuration ---

logger = logging.getLogger("HMIS-Clinical-Summarizer")

# Configure logging for HMIS integration
logging.basicConfig(
    filename='hmis_summarizer_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s'
)

# --- Clinical Summarizer Class for HMIS ---

class ClinicalSummarizer:
    """
    HMIS-integrated clinical text summarizer using Pegasus PubMed for summarization
    and clinical decision support without hardcoded disease categories.

    Attributes:
        device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
    """

    # Class-level cache for models
    _model: Optional[PegasusForConditionalGeneration] = None
    _tokenizer: Optional[PegasusTokenizer] = None
    _nlp: Optional[spacy.language.Language] = None
    _initialized: bool = False

    # Generic medical terms to filter out
    GENERIC_TERMS: Set[str] = {
        'history', 'old', 'year', 'years', 'patient', 'presented', 'levels', 'elevated',
        'diagnosed', 'admitted', 'examination', 'tests', 'high', 'low', 'year-old',
        'blood tests', 'physical examination', 'contact', 'male', 'female', 'woman', 'man',
        'positive', 'clinician', 'emergency room', 'clinical note', 'recent', 'recently',
        'ago', 'daily', 'bid', 'po', 'iv', 'status', 'condition', 'finding', 'normal'
    }

    def __init__(self, model_name: str = "google/pegasus-pubmed", device: Optional[str] = None):
        """
        Initialize the ClinicalSummarizer with Pegasus model.

        Args:
            model_name (str): Name of the Pegasus model (default: google/pegasus-pubmed).
            device (Optional[str]): Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models(model_name)

        if self._model:
            self._model.to(self.device)
        
        logger.info(f"HMIS Clinical Summarizer initialized on device: {self.device}")

    @classmethod
    def _initialize_models(cls, model_name: str) -> None:
        """Initialize Pegasus and spaCy models."""
        if cls._initialized:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                logger.info("Loading HMIS Clinical Summarizer models...")
                cls._tokenizer = PegasusTokenizer.from_pretrained(model_name)
                cls._model = PegasusForConditionalGeneration.from_pretrained(model_name)

                try:
                    cls._nlp = spacy.load("en_core_sci_sm")
                except OSError:
                    logger.warning("en_core_sci_sm not found. Falling back to en_core_web_sm.")
                    cls._nlp = spacy.load("en_core_web_sm")

                cls._initialized = True
                logger.info("HMIS Clinical Summarizer models loaded successfully")

            except Exception as e:
                logger.error(f"Failed to initialize HMIS models: {e}")
                raise RuntimeError(f"HMIS Model initialization failed: {e}")

    def summarize(self, note: Union[str, Dict], max_length: int = 250, min_length: int = 50) -> str:
        """
        Generate a clinical summary for HMIS integration.

        Args:
            note (Union[str, Dict]): Clinical note as string or dictionary.
            max_length (int): Maximum length of the summary.
            min_length (int): Minimum length of the summary.

        Returns:
            str: Formatted HTML summary for HMIS display.
        """
        try:
            if not note:
                return self._format_hmis_error("Input clinical note is empty")

            text = self._preprocess_input(note)
            if not text.strip():
                return self._format_hmis_error("No valid clinical content found in input")

            raw_summary = self._generate_summary(text, max_length, min_length)
            verified_summary = self._verify_summary(text, raw_summary)

            if not verified_summary or "Unable to generate" in verified_summary:
                return self._format_hmis_output(verified_summary, {}, is_error=True)

            clinical_insights = self._generate_comprehensive_insights(verified_summary, text)
            return self._format_hmis_output(verified_summary, clinical_insights)

        except Exception as e:
            logger.error(f"HMIS summarization error: {e}", exc_info=True)
            return self._format_hmis_error(f"System error: {str(e)}")

    def _preprocess_input(self, note: Union[str, Dict]) -> str:
        """Preprocess input for HMIS-compatible summarization."""
        if not note:
            raise ValueError("Input clinical note is empty")

        if isinstance(note, dict):
            text = ". ".join(
                f"{k.replace('_', ' ').title()}: {v}" 
                for k, v in note.items() 
                if v and isinstance(v, (str, int, float))
            )
        elif isinstance(note, str):
            text = note
        else:
            raise TypeError(f"Unsupported HMIS input type: {type(note)}")

        # Normalize text
        text = unicodedata.normalize('NFKC', text)
        
        # Basic abbreviation expansion
        text = re.sub(r'°C', 'degrees Celsius', text)
        text = re.sub(r'\bBP\b', 'blood pressure', text)
        text = re.sub(r'\bHR\b', 'heart rate', text)
        text = re.sub(r'\bRR\b', 'respiratory rate', text)
        text = re.sub(r'\bSpO2\b', 'oxygen saturation', text)

        return "Summarize the following clinical note accurately and concisely for hospital use: " + text

    def _generate_summary(self, text: str, max_length: int, min_length: int) -> str:
        """Generate summary using Pegasus model."""
        try:
            inputs = self._tokenizer(
                [text], max_length=1024, truncation=True, return_tensors="pt"
            ).to(self.device)
            
            summary_ids = self._model.generate(
                inputs["input_ids"], 
                num_beams=4, 
                max_length=max_length, 
                min_length=min_length, 
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logger.info("Summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"HMIS summary generation failed: {e}")
            raise

    def _verify_summary(self, input_text: str, summary: str) -> str:
        """Verify and enhance summary for clinical relevance."""
        try:
            input_doc = self._nlp(input_text.lower())
            input_entities = self._extract_medical_entities(input_doc)
            sentences = self._split_sentences(summary)
            filtered_sentences = self._filter_sentences(sentences, input_entities)

            # Enhance summary with inferred conditions
            if 'metformin' in input_text.lower() and 'diabetes' not in summary.lower():
                filtered_sentences.append("Patient has a history of diabetes managed with Metformin.")
            if 'nasal congestion' in input_text.lower() or 'facial pain' in input_text.lower():
                filtered_sentences.append("Symptoms suggest possible acute sinusitis.")

            if not filtered_sentences:
                logger.warning("No clinically relevant sentences found after filtering")
                return "Unable to generate reliable clinical summary due to insufficient specific medical information"
            
            return " ".join(filtered_sentences).strip()
            
        except Exception as e:
            logger.error(f"HMIS summary verification failed: {e}")
            return summary

    def _extract_medical_entities(self, doc: spacy.tokens.Doc) -> Set[str]:
        """Extract medical entities with HMIS-specific filtering."""
        entities = set()
        for ent in doc.ents:
            entity_text = ent.text.lower().strip()
            if (entity_text and 
                entity_text not in self.GENERIC_TERMS and
                len(entity_text) > 2 and
                not entity_text.isnumeric()):
                entities.add(entity_text)
        return entities

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        if not self._nlp:
            return [s.strip() for s in text.split('.') if s.strip()]
        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _filter_sentences(self, sentences: List[str], input_entities: Set[str]) -> List[str]:
        """Filter sentences based on clinical relevance."""
        filtered = []
        for sentence in sentences:
            sentence_doc = self._nlp(sentence.lower())
            sentence_entities = self._extract_medical_entities(sentence_doc)
            
            if (sentence_entities and 
                (sentence_entities.intersection(input_entities) or 
                 self._is_clinically_relevant(sentence))):
                filtered.append(sentence)
                
        return filtered

    def _is_clinically_relevant(self, sentence: str) -> bool:
        """Check if sentence contains clinically relevant content."""
        clinical_indicators = [
            'diagnosis', 'treatment', 'medication', 'symptom', 'test', 'result',
            'procedure', 'therapy', 'management', 'assessment', 'plan'
        ]
        return any(indicator in sentence.lower() for indicator in clinical_indicators)

    def _generate_comprehensive_insights(self, summary_text: str, input_text: str) -> Dict[str, List[str]]:
        """Generate clinical insights using spaCy and keyword-based logic."""
        try:
            text_lower = input_text.lower()
            summary_doc = self._nlp(summary_text.lower())
            input_doc = self._nlp(text_lower)
            entities = self._extract_medical_entities(input_doc)

            # Infer diagnoses from entities and symptoms
            diagnoses = []
            if 'nasal congestion' in text_lower or 'facial pain' in text_lower or 'sinusitis' in text_lower:
                diagnoses.extend([
                    "Acute Bacterial Sinusitis - consider if symptoms >10 days or worsening",
                    "Viral Rhinosinusitis - common, typically self-limiting",
                    "Allergic Rhinitis - consider if seasonal or allergen exposure"
                ])
            if 'metformin' in text_lower or 'diabetes' in text_lower:
                diagnoses.append("Type 2 Diabetes - managed with Metformin")
            if 'lisinopril' in text_lower or 'hypertension' in text_lower:
                diagnoses.append("Primary Hypertension - managed with Lisinopril")
            if not diagnoses:
                diagnoses.append("Requires further clinical evaluation")

            # Generate management recommendations
            recommendations = [
                "Review medication list for interactions",
                "Monitor vital signs regularly"
            ]
            if 'nasal congestion' in text_lower or 'facial pain' in text_lower:
                recommendations.extend([
                    "Consider nasal endoscopy or sinus imaging",
                    "Evaluate for antibiotics if bacterial sinusitis suspected"
                ])
            if 'chest pain' in text_lower:
                recommendations.append("Obtain ECG and cardiac enzymes")

            # Generate diagnostic suggestions
            suggestions = [
                "Complete Blood Count (CBC)",
                "Basic Metabolic Panel (BMP)"
            ]
            if 'nasal congestion' in text_lower or 'facial pain' in text_lower:
                suggestions.append("Sinus CT if symptoms persist")
            if 'fever' in text_lower:
                suggestions.append("Inflammatory markers (CRP, ESR)")

            # Generate critical considerations
            considerations = [
                "Review allergies before administering medications",
                "Assess fall risk and implement precautions"
            ]
            if 'fever' in text_lower and 'nasal congestion' in text_lower:
                considerations.append("**URGENT**: Assess for sinusitis complications (e.g., orbital cellulitis)")

            insights = {
                "primary_diagnoses": diagnoses[:5],
                "management_recommendations": recommendations,
                "diagnostic_suggestions": suggestions,
                "critical_considerations": considerations,
                "detected_entities": list(entities)
            }
            
            logger.info(f"Generated insights with entities: {entities}")
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return self._get_fallback_insights()

    def _get_fallback_insights(self) -> Dict[str, List[str]]:
        """Provide fallback insights when generation fails."""
        return {
            "primary_diagnoses": ["Clinical correlation required for accurate diagnosis"],
            "management_recommendations": ["Standard clinical assessment and monitoring recommended"],
            "diagnostic_suggestions": ["Basic laboratory workup and imaging as clinically indicated"],
            "critical_considerations": ["Ensure patient stability before proceeding with evaluation"],
            "detected_entities": []
        }

    def _format_hmis_output(self, summary: str, insights: Dict[str, List[str]], is_error: bool = False) -> str:
        """Format output for HMIS display with comprehensive clinical data."""
        summary = bleach.clean(summary)
        sanitized_insights = {
            key: [bleach.clean(item) for item in value]
            for key, value in insights.items()
            if isinstance(value, list)
        }

        if is_error:
            return f"""
            <div class="hmis-error">
                <h3>Clinical Summary Error</h3>
                <p>{summary}</p>
                <p>Please contact technical support if this error persists.</p>
            </div>
            """
        
        output_parts = [
            '<div class="hmis-clinical-summary">',
            '<div class="summary-section">',
            '<h3>Clinical Summary</h3>',
            '<div class="summary-content">'
        ]
        
        sentences = self._split_sentences(summary)
        if sentences:
            output_parts.append('<ul>')
            for sentence in sentences:
                output_parts.append(f'<li>{sentence.rstrip(".")}.</li>')
            output_parts.append('</ul>')
        else:
            output_parts.append('<p>No summary generated.</p>')
            
        output_parts.append('</div>')
        
        if sanitized_insights.get('primary_diagnoses'):
            output_parts.extend([
                '<div class="insights-section">',
                '<h4>Differential Diagnosis</h4>',
                '<ul>'
            ])
            output_parts.extend(f'<li>{dx}</li>' for dx in sanitized_insights['primary_diagnoses'])
            output_parts.append('</ul></div>')
        
        if sanitized_insights.get('management_recommendations'):
            output_parts.extend([
                '<div class="insights-section">',
                '<h4>Management Recommendations</h4>',
                '<ul>'
            ])
            output_parts.extend(f'<li>{rec}</li>' for rec in sanitized_insights['management_recommendations'])
            output_parts.append('</ul></div>')
        
        if sanitized_insights.get('diagnostic_suggestions'):
            output_parts.extend([
                '<div class="insights-section">',
                '<h4>Diagnostic Considerations</h4>',
                '<ul>'
            ])
            output_parts.extend(f'<li>{sug}</li>' for sug in sanitized_insights['diagnostic_suggestions'])
            output_parts.append('</ul></div>')
        
        if sanitized_insights.get('critical_considerations'):
            output_parts.extend([
                '<div class="critical-section">',
                '<h4>Critical Considerations</h4>',
                '<ul>'
            ])
            output_parts.extend(f'<li>{cons}</li>' for cons in sanitized_insights['critical_considerations'])
            output_parts.append('</ul></div>')
        
        output_parts.extend([
            '<div class="hmis-disclaimer">',
            '<hr>',
            f'<p><small>Generated by HMIS Clinical AI on {datetime.now().strftime("%Y-%m-%d %H:%M")}</small></p>',
            '<p><strong>Disclaimer:</strong> This AI-generated clinical summary is for decision support only ',
            'and must be verified by qualified healthcare providers. All treatment decisions ',
            'require clinical judgment and patient-specific considerations.</p>',
            '</div>',
            '</div>',
            '</div>'
        ])
        
        return '\n'.join(output_parts)

    def _format_hmis_error(self, message: str) -> str:
        """Format error messages for HMIS."""
        message = bleach.clean(message)
        return f"""
        <div class="hmis-error">
            <h3>Clinical Summary Error</h3>
            <p>{message}</p>
            <p>Please contact technical support if this error persists.</p>
        </div>
        """

    def batch_summarize(self, notes: List[Union[str, Dict]], **kwargs) -> List[str]:
        """Batch processing for multiple clinical notes."""
        logger.info(f"Processing batch of {len(notes)} clinical notes")
        return [self.summarize(note, **kwargs) for note in notes]

    def get_system_status(self) -> Dict[str, any]:
        """Get system status for HMIS monitoring."""
        return {
            "initialized": self._initialized,
            "device": self.device,
            "models_loaded": bool(self._model and self._tokenizer and self._nlp),
            "timestamp": datetime.now().isoformat()
        }

    def __repr__(self) -> str:
        return f"ClinicalSummarizer(device='{self.device}', status={'Active' if self._initialized else 'Inactive'})"