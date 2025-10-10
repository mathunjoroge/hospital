import logging
import re
import bleach
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
import json
import os
import spacy
import pdfplumber

# --- Logging Configuration ---
logging.basicConfig(
    filename='medical_chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger("MedicalChatbot")

# --- API Configuration ---
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# --- Load SpaCy for NLP Tasks ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("SpaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
    raise

# --- Universal Clinical Summarizer Class ---
class UniversalClinicalSummarizer:
    """A medical chatbot using Gemini API for detailed, clinician-focused responses."""
    
    SAFETY_FILTERS: Dict[str, Any] = {
        "dangerous_advice": [
            r"self-diagnose", r"self-treat", r"stop taking", r"ignore.*doctor",
            r"alternative to.*medical", r"without.*doctor"
        ],
        "emergency_conditions": [
            "chest pain", "difficulty breathing", "severe headache", "sudden weakness",
            "uncontrolled bleeding", "suicidal thoughts", "severe allergic reaction"
        ],
        "clinical_rules": {
            "contraindicated_beta_blockers": r"beta.*blocker.*asthma",
            "outdated_diabetes_guideline": r"insulin.*start.*hba1c.*<7"
        }
    }

    def __init__(self, gemini_api_key: str, user_type: str = "doctor"):
        """Initialize the chatbot with a Gemini API key and user type."""
        if not gemini_api_key:
            raise ValueError("Gemini API key is required to connect to the service.")
        self.gemini_api_key = gemini_api_key
        self.user_type = user_type.lower()  # "patient" or "clinician"
        logger.info(f"MedicalChatbot initialized with model: {GEMINI_MODEL}, user_type: {user_type}")

    async def _query_gemini_async(self, contents: List[Dict[str, Any]], max_tokens: int = 3000, max_retries: int = 5) -> Dict[str, Any]:
        """Query Gemini API asynchronously with Google Search grounding and exponential backoff."""
        url = f"{API_BASE_URL}/{GEMINI_MODEL}:generateContent?key={self.gemini_api_key}"
        
        # Detect specialty from the latest user query
        specialty = self.detect_specialty(contents[-1]["parts"][0]["text"]) if contents else "general"
        
        system_instruction = f"""
        You are an **evidence-based clinical AI assistant** designed to support clinicians with precise, technical, and comprehensive medical information. Your purpose is to assist doctors by providing detailed clinical insights, including pathophysiology, differential diagnoses, diagnostic workup, evidence-based management, and patient counseling strategies.

        ### CORE DIRECTIVES
        1. **Prioritize High-Quality Sources**: Ground responses in peer-reviewed journals (e.g., NEJM, Lancet, JAMA), clinical guidelines (e.g., UpToDate, NICE, AHA/ACC), and authoritative medical databases (e.g., PubMed, Cochrane). Avoid general web sources unless from reputable organizations (e.g., WHO, CDC, NIH).
        2. **Apply Clinical Decision-Making Frameworks**: Structure responses to include:
           - **Overview**: Brief summary of the condition or query.
           - **Pathophysiology**: Explain the underlying mechanisms.
           - **Differential Diagnosis**: List and prioritize potential diagnoses based on provided symptoms or context.
           - **Diagnostic Workup**: Recommend specific tests (e.g., labs, imaging, procedures) with rationale.
           - **Management**: Provide evidence-based treatment options, including medications (with dosages where applicable), non-pharmacologic interventions, and follow-up plans.
           - **Patient Counseling**: Suggest key points for patient education.
           - **References**: Cite sources (e.g., [NEJM, 2023, DOI: xxx]).
        3. **Use Specialty-Specific Terminology**: Tailor language to the relevant medical specialty (e.g., {specialty}) based on the query context.
        4. **Cite Sources Explicitly**: Include inline citations and a reference list with DOIs or URLs.
        5. **Handle Ambiguity**: If the query is vague, ask clarifying questions (e.g., 'Can you specify the patient's age, symptoms, or medical history?') and provide a broad differential diagnosis.
        6. **Maintain Scientific Rigor**: Avoid speculation and ensure all recommendations align with current (as of October 2025) clinical guidelines.

        ### STYLE AND TONE
        - Use precise, technical medical terminology suitable for clinicians.
        - Structure responses with clear headings and bullet points.
        - Be concise yet comprehensive, avoiding unnecessary elaboration.
        - Always include a disclaimer: 'This information is for educational purposes only and not a substitute for clinical judgment. Consult authoritative sources and patient-specific data before making decisions.'

        ### RESPONSE TEMPLATE
        - **Overview**: [Summary]
        - **Pathophysiology**: [Mechanisms]
        - **Differential Diagnosis**: [List with likelihood]
        - **Diagnostic Workup**: [Tests and rationale]
        - **Management**: [Treatments, dosages, interventions]
        - **Patient Counseling**: [Education points]
        - **References**: [Citations]

        **NOTE**: Responses must be complete, accurate, and never truncated. If the query involves a medical emergency, prioritize urgent action recommendations.
        """
        
        if specialty != "general":
            system_instruction += f"\nTailor the response for a {specialty} specialist, using relevant terminology and focusing on specialty-specific guidelines."

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.5 if self.user_type == "doctor" else 0.3,
            },
            "tools": [{"google_search": {}}],
        }
        
        headers = {'Content-Type': 'application/json'}
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                        response.raise_for_status()
                        result = await response.json()
                        
                        if 'promptFeedback' in result and result['promptFeedback'].get('blockReason'):
                            logger.warning(f"Response blocked: {result['promptFeedback']['blockReason']}")
                            return {"text": "", "sources": []}

                        if not result.get('candidates'):
                            if attempt < max_retries - 1:
                                logger.warning("API returned no candidates. Retrying...")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            raise ValueError("API response structure is invalid or candidates are missing after retries.")

                        candidate = result['candidates'][0]
                        text = candidate['content']['parts'][0]['text'].strip()
                        sources: List[Dict[str, str]] = []
                        
                        grounding_metadata = candidate.get('groundingMetadata')
                        if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                            for attribution in grounding_metadata['groundingAttributions']:
                                if 'web' in attribution:
                                    sources.append({
                                        "uri": attribution['web'].get('uri', 'N/A'),
                                        "title": attribution['web'].get('title', 'N/A')
                                    })
                        
                        logger.info(f"API response received. Length: {len(text)} characters")
                        return {"text": text, "sources": sources}

                except Exception as e:
                    logger.warning(f"API Request failed (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error("Max retries reached. Failing the API call.")
                        return {"text": "", "sources": []}
        
        return {"text": "", "sources": []}

    def _query_gemini(self, contents: List[Dict[str, Any]], max_tokens: int = 3000, max_retries: int = 5) -> Dict[str, Any]:
        """Synchronous wrapper for async Gemini query."""
        return asyncio.run(self._query_gemini_async(contents, max_tokens, max_retries))

    def detect_specialty(self, question: str) -> str:
        """Detect the medical specialty from the question using SpaCy."""
        doc = nlp(question.lower())
        specialty_keywords = {
            "cardiology": ["heart", "chest pain", "arrhythmia", "hypertension"],
            "neurology": ["headache", "seizure", "stroke", "numbness"],
            "endocrinology": ["diabetes", "thyroid", "hba1c", "hormone"],
            "pulmonology": ["cough", "asthma", "copd", "shortness of breath"],
            "gastroenterology": ["abdominal pain", "diarrhea", "nausea", "ulcer"],
        }
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in doc.text for keyword in keywords):
                return specialty
        return "general"

    def parse_clinical_data(self, clinical_data: Dict[str, Any]) -> str:
        """Parse structured clinical data (e.g., labs, vitals) to include in the query context."""
        try:
            vitals = clinical_data.get('vitals', {})
            labs = clinical_data.get('labs', {})
            history = clinical_data.get('history', '')

            context = f"Patient Context:\n"
            if vitals:
                context += f"- Vitals: {', '.join([f'{k}: {v}' for k, v in vitals.items()])}\n"
            if labs:
                context += f"- Labs: {', '.join([f'{k}: {v}' for k, v in labs.items()])}\n"
            if history:
                context += f"- Medical History: {history}\n"
            
            logger.info("Parsed clinical data successfully.")
            return context
        except Exception as e:
            logger.error(f"Error parsing clinical data: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
            logger.info(f"Extracted text from PDF: {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return ""

    def _check_emergency(self, question: str) -> Optional[str]:
        """Check if the question indicates a medical emergency using NLP."""
        doc = nlp(question.lower())
        emergency_entities = [ent.text for ent in doc.ents if ent.label_ in ["SYMPTOM", "CONDITION"]]
        for condition in self.SAFETY_FILTERS["emergency_conditions"]:
            if any(condition in entity for entity in emergency_entities):
                return self._format_emergency_response(condition)
        return None

    def _format_emergency_response(self, condition: str) -> str:
        """Format emergency response with clear instructions."""
        emergency_responses = {
            "chest pain": "🚨 **EMERGENCY WARNING** 🚨 This could indicate a life-threatening event like a heart attack. **CALL EMERGENCY SERVICES (e.g., 911/112) IMMEDIATELY** or go to the nearest emergency room. Do not wait.",
            "difficulty breathing": "🚨 **EMERGENCY WARNING** 🚨 Difficulty breathing is a medical emergency. **CALL EMERGENCY SERVICES IMMEDIATELY.**",
            "severe headache": "🚨 **EMERGENCY WARNING** 🚨 A sudden, severe headache could indicate a serious neurological issue. **SEEK EMERGENCY CARE IMMEDIATELY.**",
            "sudden weakness": "🚨 **EMERGENCY WARNING** 🚨 This could indicate a stroke. **CALL EMERGENCY SERVICES IMMEDIATELY.**",
            "uncontrolled bleeding": "🚨 **EMERGENCY WARNING** 🚨 Apply direct pressure and **SEEK EMERGENCY CARE IMMEDIATELY.**",
            "suicidal thoughts": "🚨 **EMERGENCY WARNING** 🚨 If you are having suicidal thoughts, contact a crisis helpline or **CALL EMERGENCY SERVICES IMMEDIATELY.**",
            "severe allergic reaction": "🚨 **EMERGENCY WARNING** 🚨 Use an epinephrine auto-injector if available and **SEEK EMERGENCY CARE IMMEDIATELY.**"
        }
        return emergency_responses.get(
            condition, "🚨 **MEDICAL EMERGENCY** 🚨 Seek immediate medical attention or call emergency services."
        )

    def _safe_fallback_response(self, question: str) -> str:
        """Provide a safe fallback response."""
        return f"""
I am currently unable to provide a detailed, grounded response for: "{question}"

For personalized, accurate, and safe medical information, you must:
1. Consult a qualified healthcare provider or physician.
2. Contact official medical organizations (e.g., CDC, WHO, local health authority).
3. If this is an emergency, call emergency services immediately.

This chatbot provides general, educational information only.
"""

    def _verify_response(self, response: str) -> str:
        """Verify response for safety and clinical accuracy."""
        response_lower = response.lower()

        # Check for dangerous advice
        for pattern in self.SAFETY_FILTERS["dangerous_advice"]:
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.warning(f"Dangerous advice pattern detected: {pattern}")
                return "I cannot provide advice on self-treatment or altering prescribed medications. Consult a physician."

        # Check for clinical inaccuracies
        for rule, pattern in self.SAFETY_FILTERS["clinical_rules"].items():
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.warning(f"Clinical inaccuracy detected: {rule}")
                return f"Response may contain outdated or incorrect clinical advice. Please refer to current guidelines (e.g., ADA, AHA) or consult a specialist."

        return response

    def answer(self, question: str, conversation_history: List[Dict[str, str]] = None, clinical_data: Dict[str, Any] = None, pdf_path: str = None) -> str:
        """Answer a medical question with optional clinical data or PDF input."""
        try:
            if not question or not question.strip():
                return self._format_output(
                    "Hello! I'm here to help with medical questions. What would you like to know?", 
                    question="Greeting", 
                    sources=[]
                )
            
            emergency_response = self._check_emergency(question)
            if emergency_response:
                return self._format_output(emergency_response, question=question, is_error=True)
            
            # Build the native contents list
            llm_contents = []
            if conversation_history:
                for entry in conversation_history:
                    role = entry.get('role', 'user')
                    content = entry.get('content', '')
                    if content and role in ['user', 'model']:
                        llm_contents.append({
                            "role": role,
                            "parts": [{"text": content}]
                        })
                    else:
                        logger.warning(f"Skipping invalid history entry: {entry}")

            # Add clinical data or PDF context
            if clinical_data or pdf_path:
                context = ""
                if clinical_data:
                    context += self.parse_clinical_data(clinical_data)
                if pdf_path and os.path.exists(pdf_path):
                    context += f"Medical Record Context:\n{self.extract_text_from_pdf(pdf_path)}\n"
                if context:
                    question = f"{context}\nQuestion: {question}"

            llm_contents.append({
                "role": "user",
                "parts": [{"text": question}]
            })
            
            logger.info(f"Sending {len(llm_contents)} total history entries to Gemini.")
            
            api_result = self._query_gemini(llm_contents, max_tokens=3000)
            response = api_result['text']
            sources = api_result['sources']

            if not response:
                logger.warning("Gemini API returned empty response after retries.")
                return self._format_output(self._safe_fallback_response(question), question=question, sources=[])
            
            logger.info(f"Successfully generated response of {len(response)} characters")
            response = self._verify_response(response)
            
            if "disclaimer" not in response.lower() and "consult" not in response.lower():
                response += "\n\n---"
                response += "\n**MANDATORY DISCLAIMER:** \n This information is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any health concerns or before starting a new treatment."
            
            return self._format_output(response, question=question, sources=sources)
        
        except Exception as e:
            logger.error(f"Critical error processing question: {e}")
            return self._format_output(self._safe_fallback_response(question), question=question, sources=[])

    def _format_output(self, response: str, question: str = None, sources: List[Dict[str, str]] = None, is_error: bool = False) -> str:
        """Format the response as beautiful, safety-focused HTML."""
        response_clean = bleach.clean(response)
        source_html = ""
        if sources:
            source_list = "".join(
                f'<li><a href="{bleach.clean(s["uri"])}" target="_blank" rel="noopener noreferrer">{bleach.clean(s["title"])}</a></li>'
                for s in sources
            )
            source_html = f"""
            <div class="grounding-sources">
                <h4>📚 Sources & References</h4>
                <ul>{source_list}</ul>
            </div>
            """

        if is_error:
            return f"""
            <div class="chatbot-error hmis-response">
                <h3>❌ Immediate Action Required</h3>
                <p>{response_clean}</p>
            </div>
            """
        
        display_question = bleach.clean(question) if question else "Starting our conversation"
        
        html = f"""
        <div class="container">
            <div class="response-header">
                <h3>🩺 Medical Assistant</h3>
                <p><strong>Your question:</strong> {display_question}</p>
            </div>
            <div class="response-content">
                <p>{response_clean}</p>
            </div>
            {source_html}
            <div class="disclaimer">
                <p><em>Response generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}</em></p>
                <p><em>Powered by Gemini AI with medical source verification</em></p>
            </div>
        </div>
        """
        return '\n'.join(line.strip() for line in html.split('\n'))