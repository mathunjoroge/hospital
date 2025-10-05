import logging
import re
import bleach
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
import json
import os

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

# --- Universal Clinical Summarizer Class ---
class UniversalClinicalSummarizer:
    """A medical chatbot using Gemini API for accurate, safe, and grounded medical responses."""
    
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

    def __init__(self, gemini_api_key: str):
        """Initialize the chatbot with a Gemini API key."""
        if not gemini_api_key:
            raise ValueError("Gemini API key is required to connect to the service.")
        self.gemini_api_key = gemini_api_key
        logger.info(f"MedicalChatbot initialized with model: {GEMINI_MODEL}")

    def _query_gemini(self, prompt: str, max_tokens: int = 1000, max_retries: int = 5) -> Dict[str, Any]:
        """
        Query Gemini API with Google Search grounding and exponential backoff.
        Returns a dictionary containing the 'text' and 'sources' (list of dicts).
        """
        url = f"{API_BASE_URL}/{GEMINI_MODEL}:generateContent?key={self.gemini_api_key}"
        
        system_instruction = """
        You are an evidence-based medical AI assistant designed for natural, multi-turn conversations. Your primary purpose is to educate, inform, and support users by explaining medical concepts, clinical findings, and treatment options in clear, professional language.

        Your core directives are:
        1.  **Maintain Conversational Flow:** Understand and respond to multi-part questions (e.g., "what is X, how is it treated, how can I prevent it?"). Address all parts of a user's request in a single, comprehensive, and well-structured response. Use the provided conversation history to maintain context and ensure continuity.
        2.  **Provide Structured Answers:** Organize information clearly using headings, bolding, and lists to make complex topics easy to understand.
        3.  **Ground All Information:** All factual claims must be grounded in credible, verifiable medical sources.
        4.  **Emphasize Disclaimers:** Always include a clear, prominent disclaimer that you are **not a substitute for professional medical advice, diagnosis, or treatment**.
        5.  **Encourage Professional Consultation:** Strongly encourage users to **consult a qualified healthcare professional** for personalized advice.
        6.  **Maintain Tone:** Be neutral, evidence-based, and empathetic.

        Your goal is to assist in medical understanding—not to make clinical judgments or replace professional expertise.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2, 
            },
            "tools": [{"google_search": {}}],
        }
        
        headers = {'Content-Type': 'application/json'}
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=20)
                response.raise_for_status()
                
                result = response.json()
                
                if 'promptFeedback' in result and result['promptFeedback'].get('blockReason'):
                    logger.warning(f"Response blocked: {result['promptFeedback']['blockReason']}")
                    return {"text": "", "sources": []}

                if not result.get('candidates'):
                    raise ValueError("API response structure is invalid or candidates are missing.")

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

    def answer(self, question: str, conversation_history: List[Dict[str, str]] = None) -> str:
            """Answer a medical question with accuracy, using conversation history for context.
            
            This updated function cleans previous Assistant responses in the history 
            to ensure the AI maintains conversational context without being confused 
            by HTML, disclaimers, and metadata."""
            
            try:
                if not question or not question.strip():
                    return self._format_output("Please provide a valid medical question.", sources=[])
                
                # Check for emergency conditions
                emergency_response = self._check_emergency(question)
                if emergency_response:
                    return self._format_output(emergency_response, is_error=True)
                
                # Build prompt with streamlined conversation history
                prompt = ""
                if conversation_history:
                    prompt += "Conversation History:\n"
                    
                    # Iterate over the last 5 exchanges to build context
                    for entry in conversation_history[-5:]:
                        
                        # --- START FIX: Robust Cleaning of Assistant's previous response ---
                        
                        # 1. Strip all HTML tags entirely (leaving only text)
                        clean_response = bleach.clean(entry['response'], tags=[], strip=True)
                        
                        # 2. Use a single, powerful regex to remove all non-core content:
                        #    - MANDATORY DISCLAIMER block
                        #    - Generated by / Powered by lines
                        #    - Any remaining horizontal rules (---)
                        #    - The specific in-response disclaimer text if it was included in the content
                        # The pattern matches any text starting from the word DISCLAIMER: to the end of the string,
                        # ensuring a clean cut-off.
                        
                        # We will first try to find the starting point of the noise block
                        disclaimer_start_index = clean_response.find("MANDATORY DISCLAIMER:")
                        if disclaimer_start_index != -1:
                            # Cut the string off before the noise starts
                            clean_response = clean_response[:disclaimer_start_index].strip()
                        
                        # Also remove the internal disclaimer paragraph if it's still there
                        internal_disclaimer = "Disclaimer: I am an AI assistant and cannot provide medical advice. This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for personalized advice regarding your health."
                        clean_response = clean_response.replace(internal_disclaimer, '').strip()

                        # Finally, remove the surrounding header/footer content that isn't the body
                        clean_response = re.sub(r'🩺 Grounded Medical Assistant|Question:.*|Generated by Medical Chatbot.*|---', '', clean_response, flags=re.DOTALL).strip()
                        
                        # Replace multiple newlines/spaces with a single space for neatness
                        clean_response = re.sub(r'\s+', ' ', clean_response).strip()
                        
                        # --- END FIX ---

                        # Add the cleaned history to the prompt
                        # If the question was blank (e.g. from an initial blank load), skip it
                        if entry['question'].strip():
                            prompt += f"User: {entry['question']}\nAssistant: {clean_response}\n"
                        
                    prompt += "\nCurrent Question:\n"
                prompt += question

                # Query Gemini API with grounding
                api_result = self._query_gemini(prompt)
                response = api_result['text']
                sources = api_result['sources']

                if not response:
                    logger.warning("Gemini API returned empty response after retries.")
                    return self._format_output(self._safe_fallback_response(question), sources=[])
                
                # Verify response for safety
                response = self._verify_response(response)
                
                # Add mandatory disclaimers and metadata (as before)
                response += "\n\n---"
                response += "\n**MANDATORY DISCLAIMER:** This information is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any health concerns or before starting a new treatment."
                
                return self._format_output(response, question=question, sources=sources)
            
            except Exception as e:
                logger.error(f"Critical error processing question: {e}")
                return self._format_output(self._safe_fallback_response(question), sources=[])

    def _verify_response(self, response: str) -> str:
        """Verify response for safety (e.g., preventing dangerous advice)."""
        response_lower = response.lower()
        for pattern in self.SAFETY_FILTERS["dangerous_advice"]:
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.warning(f"Dangerous advice pattern detected and will be overridden: {pattern}")
                return "Caution: I cannot provide advice on self-treatment or altering prescribed medications. You must consult your physician or a pharmacist immediately."
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
                    <style>
            .hmis-response {{
                font-family: 'Inter', sans-serif;
                background-color: #f9fafb;
                padding: 1.5rem;
                border-radius: 0.75rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
                max-width: 100%;
                margin: 1rem auto;
            }}
            .response-header h3 {{
                font-size: 1.5rem;
                font-weight: 700;
                color: #059669;
                margin-bottom: 0.5rem;
            }}
            .response-header p {{
                font-style: italic;
                color: #4b5563;
                margin-bottom: 1rem;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 0.5rem;
            }}
            .response-content p {{
                white-space: pre-wrap;
                font-size: 1rem;
                line-height: 1.6;
                color: #1f2937;
            }}
            .disclaimer {{
                margin-top: 1.5rem;
                padding-top: 1rem;
                border-top: 1px solid #e5e7eb;
                font-size: 0.875rem;
                color: #6b7280;
            }}
            .disclaimer strong {{
                color: #ef4444;
                font-weight: 600;
            }}
            .grounding-sources {{
                margin-top: 1rem;
                padding: 0.75rem;
                background-color: #e0f2f1;
                border-radius: 0.5rem;
                border: 1px solid #99f6e4;
            }}
            .grounding-sources h4 {{
                font-size: 1rem;
                color: #047857;
                margin-bottom: 0.5rem;
            }}
            .grounding-sources ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            .grounding-sources li {{
                margin-bottom: 0.25rem;
                font-size: 0.85rem;
            }}
            .grounding-sources a {{
                color: #059669;
                text-decoration: underline;
            }}
            .chatbot-error {{
                background-color: #fee2e2;
                border: 2px solid #ef4444;
            }}
            .chatbot-error h3 {{
                color: #dc2626;
            }}
        </style>
            <div class="grounding-sources">
                <h4>Sources (Grounding)</h4>
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
        
        html = f"""
        
        <div class="hmis-response">
            <div class="response-header">
                <h3>🩺 Grounded Medical Assistant</h3>
                <p><strong>Question:</strong> {bleach.clean(question) if question else "General inquiry"}</p>
            </div>
            <div class="response-content">
                <p>{response_clean}</p>
            </div>
            {source_html}
            <div class="disclaimer">
                <p><em>Generated by Medical Chatbot on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                <p><em>Powered by Gemini API</em></p>
            </div>
        </div>
        """
        return '\n'.join(line.strip() for line in html.split('\n'))