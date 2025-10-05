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

    def __init__(self, gemini_api_key: str):
        """Initialize the chatbot with a Gemini API key."""
        if not gemini_api_key:
            raise ValueError("Gemini API key is required to connect to the service.")
        self.gemini_api_key = gemini_api_key
        logger.info(f"MedicalChatbot initialized with model: {GEMINI_MODEL}")

    def _query_gemini(self, prompt: str, max_tokens: int = 1500, max_retries: int = 5) -> Dict[str, Any]:
        """
        Query Gemini API with Google Search grounding and exponential backoff.
        Returns a dictionary containing the 'text' and 'sources' (list of dicts).
        """
        url = f"{API_BASE_URL}/{GEMINI_MODEL}:generateContent?key={self.gemini_api_key}"
        
        system_instruction = """
        You are an evidence-based medical AI assistant designed for natural, multi-turn conversations. Your primary purpose is to educate, inform, and support users by explaining medical concepts, clinical findings, and treatment options in clear, professional language.

        CRITICAL DIRECTIVES:
        1. **ALWAYS provide complete, detailed medical explanations** - never cut off responses prematurely
        2. **Maintain Conversational Flow:** Understand and respond to multi-part questions and follow-up questions naturally
        3. **Be Conversational:** Use natural language, appropriate greetings, and follow-up questions
        4. **Provide Structured Answers:** Organize information clearly using headings, bolding, and lists
        5. **Ground All Information:** All factual claims must be grounded in credible, verifiable medical sources
        6. **Include Disclaimers:** Always include appropriate medical disclaimers
        7. **Maintain Tone:** Be neutral, evidence-based, and empathetic

        IMPORTANT: Your responses must be COMPLETE and not truncated. Provide full explanations.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3,
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

    def _build_conversational_prompt(self, question: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Build a conversational prompt that intelligently maintains context."""
        
        # Start with strong context instructions
        prompt = """You are having a natural, ongoing conversation about medical topics. You MUST maintain context from the conversation history.

CRITICAL RULES FOR FOLLOW-UP QUESTIONS:
1. When a user asks a short follow-up question like "what are the causes", "tell me more", "how is it treated", etc., you MUST assume it refers to the most recent topic discussed
2. NEVER ask for clarification on follow-up questions - use the context from our previous discussion
3. Continue the conversation naturally about the same topic
4. Provide detailed, specific information relevant to the ongoing discussion"""
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt += "\n\nCONVERSATION HISTORY (MOST RECENT FIRST):\n"
            # Show the most recent exchanges first for better context
            for i, entry in enumerate(reversed(conversation_history[-3:])):
                clean_response = self._clean_previous_response(entry['response'])
                if entry['question'].strip() and clean_response.strip():
                    prompt += f"USER: {entry['question']}\n"
                    prompt += f"YOU: {clean_response}\n\n"
        
        # Enhanced follow-up detection with stronger context
        is_follow_up = self._is_follow_up_question(question, conversation_history)
        
        if is_follow_up and conversation_history and len(conversation_history) > 0:
            # Get the most recent conversation for context
            last_exchange = conversation_history[-1]
            last_question = last_exchange['question']
            last_response_clean = self._clean_previous_response(last_exchange['response'])
            
            prompt += f"CONTEXT FOR CURRENT QUESTION:\n"
            prompt += f"We are currently discussing: '{last_question}'\n"
            prompt += f"Your last response was about: {last_response_clean[:300]}...\n"
            prompt += f"The user's current question '{question}' is a DIRECT FOLLOW-UP about this same topic.\n\n"
            
            prompt += "RESPONSE REQUIREMENTS:\n"
            prompt += "✅ Continue discussing the SAME TOPIC from our previous conversation\n"
            prompt += "✅ DO NOT ask 'what do you mean?' or ask for clarification\n"
            prompt += "✅ Provide specific, detailed information relevant to the ongoing discussion\n"
            prompt += "✅ Structure your response clearly\n"
            prompt += "✅ Include appropriate medical disclaimers\n\n"
            
            prompt += f"Based on our discussion about '{last_question}', please answer the follow-up question: '{question}'\n\n"
        
        prompt += f"CURRENT USER MESSAGE: {question}\n\n"
        
        if not is_follow_up:
            prompt += """Please provide a complete, detailed explanation that:
1. Answers the question thoroughly
2. Uses clear structure with headings and bullet points if helpful
3. Includes relevant medical information
4. Adds appropriate disclaimers
5. Is conversational and helpful

Response:"""
        
        return prompt

    def _is_follow_up_question(self, current_question: str, conversation_history: List[Dict[str, str]] = None) -> bool:
        """Determine if the current question is likely a follow-up to previous conversation."""
        if not conversation_history or len(conversation_history) == 0:
            return False
        
        follow_up_indicators = [
            'what are the', 'how do you', 'can you explain', 'tell me more',
            'what about', 'and what', 'how about', 'what causes',
            'what symptoms', 'how is it', 'what treatment', 'how do they',
            'what is the', 'can it be', 'is it', 'does it', 'what is it',
            'what was that', 'explain more', 'go on', 'continue', 'and how',
            'what about the', 'tell me about', 'what else', 'how about',
            'what are', 'how are', 'why are', 'when are', 'where are'
        ]
        
        current_lower = current_question.lower().strip()
        
        # Check if it contains follow-up phrases
        for indicator in follow_up_indicators:
            if current_lower.startswith(indicator) or f" {indicator} " in f" {current_lower} ":
                return True
        
        # Check if it's a short, context-dependent question (5 words or less)
        if len(current_question.split()) <= 5:
            return True
        
        # Check for pronouns that reference previous content
        reference_pronouns = ['they', 'it', 'that', 'this', 'those', 'these']
        if any(pronoun in current_lower.split() for pronoun in reference_pronouns):
            return True
        
        # Check for very short questions that are likely follow-ups
        if len(current_question) <= 20 and current_lower not in ['hello', 'hi', 'help', 'thanks', 'thank you']:
            return True
        
        return False

    def _clean_previous_response(self, response: str) -> str:
        """Clean previous responses to remove HTML and formatting for context."""
        if not response:
            return ""
            
        # Remove HTML tags
        clean_text = bleach.clean(response, tags=[], strip=True)
        
        # Remove the disclaimer section and footer
        disclaimer_start = clean_text.find("MANDATORY DISCLAIMER:")
        if disclaimer_start != -1:
            clean_text = clean_text[:disclaimer_start].strip()
        
        # Remove header information and metadata
        clean_text = re.sub(r'🩺 Medical Assistant.*?Your question:', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'Response generated on.*', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'Powered by Gemini AI.*', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'---.*', '', clean_text, flags=re.DOTALL)
        
        # Clean up extra whitespace but preserve paragraph structure
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        return clean_text

    def answer(self, question: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Answer a medical question with natural conversation flow."""
        try:
            if not question or not question.strip():
                return self._format_output(
                    "Hello! I'm here to help with medical questions. What would you like to know?", 
                    question="Greeting", 
                    sources=[]
                )
            
            # Check for emergency conditions
            emergency_response = self._check_emergency(question)
            if emergency_response:
                return self._format_output(emergency_response, question=question, is_error=True)
            
            # Build conversational prompt with improved context handling
            prompt = self._build_conversational_prompt(question, conversation_history)
            
            logger.info(f"Generated prompt for question: {question}")
            logger.info(f"Conversation history length: {len(conversation_history) if conversation_history else 0}")
            
            # Query Gemini API with grounding - increased tokens for complete responses
            api_result = self._query_gemini(prompt, max_tokens=1800)
            response = api_result['text']
            sources = api_result['sources']

            if not response:
                logger.warning("Gemini API returned empty response after retries.")
                return self._format_output(self._safe_fallback_response(question), question=question, sources=[])
            
            logger.info(f"Successfully generated response of {len(response)} characters")
            
            # Verify response for safety
            response = self._verify_response(response)
            
            # Add mandatory disclaimers and metadata
            if "disclaimer" not in response.lower() and "consult" not in response.lower():
                response += "\n\n---"
                response += "\n**MANDATORY DISCLAIMER:** \n This information is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any health concerns or before starting a new treatment."
            
            return self._format_output(response, question=question, sources=sources)
        
        except Exception as e:
            logger.error(f"Critical error processing question: {e}")
            return self._format_output(self._safe_fallback_response(question), question=question, sources=[])

    def _verify_response(self, response: str) -> str:
        """Verify response for safety (e.g., preventing dangerous advice)."""
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
        
        # Format question for display (handle empty questions for initial state)
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