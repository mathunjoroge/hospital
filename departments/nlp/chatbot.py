import logging
import re
import bleach
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
import json
import os
import spacy
import numpy as np
from scipy.spatial.distance import cosine

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

# --- Load SpaCy for Semantic Similarity ---
# NOTE: While spacy is no longer used for conversational logic, 
# it's kept here in case it's needed for other NLP tasks.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("SpaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
    raise

# --- Universal Clinical Summarizer Class ---
class UniversalClinicalSummarizer:
    """A medical chatbot using Gemini API for natural, conversational medical responses."""
    
    SAFETY_FILTERS: Dict[str, Any] = {
        "dangerous_advice": [
            r"self-diagnose", r"self-treat", r"stop taking", r"ignore.*doctor",
            r"alternative to.*medical", r"without.*doctor"
        ],
        "emergency_conditions": [
            "chest pain", "difficulty breathing", "severe headache", "sudden weakness",
            "uncontrolled bleeding", "suicidal thoughts", "severe allergic reaction"
        ]
    }

    def __init__(self, gemini_api_key: str, user_type: str = "doctor"):
        """Initialize the chatbot with a Gemini API key and user type."""
        if not gemini_api_key:
            raise ValueError("Gemini API key is required to connect to the service.")
        self.gemini_api_key = gemini_api_key
        self.user_type = user_type.lower()  # "patient" or "clinician"
        logger.info(f"MedicalChatbot initialized with model: {GEMINI_MODEL}, user_type: {user_type}")

    # FIX 1: _query_gemini now accepts the full 'contents' list (conversation history)
    def _query_gemini(self, contents: List[Dict[str, Any]], max_tokens: int = 1500, max_retries: int = 5) -> Dict[str, Any]:
        """Query Gemini API with Google Search grounding and exponential backoff."""
        url = f"{API_BASE_URL}/{GEMINI_MODEL}:generateContent?key={self.gemini_api_key}"
        
        system_instruction = f"""
        You are an **evidence-based clinical AI assistant** trained to provide accurate, comprehensive, and context-aware medical information through natural, multi-turn conversations. Your purpose is to **educate, inform, and support** users — {'patients with clear, empathetic language' if self.user_type == 'patient' else 'clinicians with precise, technical details'} — by explaining medical concepts, findings, and management options.

        ### CORE DIRECTIVES
        1. **Provide Comprehensive, Evidence-Based Explanations:** Always deliver thorough, accurate, and up-to-date medical information grounded in reputable clinical sources (e.g., WHO, CDC, NICE, NIH, peer-reviewed journals).

        2. **Ensure Conversational and Contextual Understanding:** Recognize and respond effectively to multi-part or follow-up questions, maintaining a coherent conversational flow.

        3. **Communicate Professionally and Empathetically:** {'Use simple, empathetic language for patients, avoiding jargon unless requested.' if self.user_type == 'patient' else 'Use precise, technical language suitable for clinicians, including medical terminology.'}

        4. **Organize Responses Logically:** Structure answers with clear headings, bullet points, and concise summaries where appropriate.

        5. **Maintain Scientific Integrity:** Avoid speculation, misinformation, or unsupported claims. Cite or reference authoritative sources when applicable.

        6. **Apply Clinical Reasoning Principles:** Where relevant, discuss differential diagnoses, pathophysiology, diagnostic workup, management strategies, and patient counseling.

        7. **Include Appropriate Medical Disclaimers:** Always clarify that your guidance is for **educational and informational purposes only**, and not a substitute for professional medical advice, diagnosis, or treatment.

        ### STYLE AND TONE
        - Be **neutral, evidence-driven, and empathetic**.  
        - Avoid sensational or absolute language.  
        - Encourage users to consult qualified healthcare professionals for personalized medical care.

        **NOTE:** Responses must be **complete**, **well-structured**, and **never truncated**. Always deliver full explanations with professionalism and precision.
        """

        payload = {
            # FIX 1.1: Use the incoming contents list directly
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3 if self.user_type == "patient" else 0.5,
            },
            "tools": [{"google_search": {}}],
        }
        
        headers = {'Content-Type': 'application/json'}
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                
                if 'promptFeedback' in result and result['promptFeedback'].get('blockReason'):
                    logger.warning(f"Response blocked: {result['promptFeedback']['blockReason']}")
                    return {"text": "", "sources": []}

                if not result.get('candidates'):
                    # The model might return a response without candidates if the history is too long or complex.
                    # This check is vital.
                    if attempt < max_retries - 1:
                        logger.warning("API returned no candidates. Retrying...")
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue # Continue to the next attempt
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

            except requests.exceptions.RequestException as e:
                logger.warning(f"API Request failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Failing the API call.")
                    return {"text": "", "sources": []}
            except Exception as e:
                logger.error(f"Error processing API response: {e}")
                return {"text": "", "sources": []}
        
        return {"text": "", "sources": []}

    # --- REMOVED HELPER FUNCTIONS ---
    # The following functions are removed as they are redundant and interfere with 
    # the LLM's native context handling:
    # - _summarize_conversation_history
    # - _build_conversational_prompt
    # - _is_follow_up_question
    # - _clean_previous_response

    def _check_emergency(self, question: str) -> Optional[str]:
        """Check if the question indicates a medical emergency."""
        question_lower = question.lower()
        for condition in self.SAFETY_FILTERS["emergency_conditions"]:
            if condition in question_lower:
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
    
    # FIX 2: Refactored 'answer' to build the native 'contents' history format
    def answer(self, question: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Answer a medical question using native chat history for natural conversation flow."""
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
            
            # --- Build the native contents list ---
            llm_contents = []
            if conversation_history:
                for entry in conversation_history:
                    # FIX 2.1: Assumes app.py now saves history in the format: 
                    # {'role': 'user'/'model', 'content': 'clean text response'}
                    role = entry.get('role', 'user') # Default to user for safety
                    content = entry.get('content', '')

                    if content and role in ['user', 'model']:
                        llm_contents.append({
                            "role": role,
                            "parts": [{"text": content}]
                        })
                    else:
                        logger.warning(f"Skipping invalid history entry: {entry}")

            # FIX 2.2: Add the current user question
            llm_contents.append({
                "role": "user",
                "parts": [{"text": question}]
            })
            
            logger.info(f"Sending {len(llm_contents)} total history entries to Gemini.")
            
            # FIX 2.3: Query the updated Gemini function using the contents list
            api_result = self._query_gemini(llm_contents, max_tokens=1800)
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

    def _verify_response(self, response: str) -> str:
        """Verify response for safety."""
        response_lower = response.lower()
        for pattern in self.SAFETY_FILTERS["dangerous_advice"]:
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.warning(f"Dangerous advice pattern detected and will be overridden: {pattern}")
                return "I understand you're looking for guidance, but I cannot provide advice on self-treatment or altering prescribed medications. It's essential to consult your physician or a pharmacist for personalized medical advice."
        return response

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