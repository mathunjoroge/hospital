import logging
import os
import re
import bleach
from flask import current_app
from extensions import db

try:
    from celery import shared_task
except ImportError:
    def shared_task(func=None, **kwargs):
        def decorator(f):
            def delay_impl(*a, **kw):
                return f(*a, **kw)
            f.delay = delay_impl
            return f
        if callable(func):
            return decorator(func)
        return decorator

from departments.api.ai_audit import AIMode, AITimer, log_ai_call
from departments.nlp.chatbot import UniversalClinicalSummarizer

logger = logging.getLogger(__name__)


@shared_task
def process_clinical_chatbot_task(combined_input: str, conversation_context: list = None, patient_id: str = None):
    """
    Celery task to run UniversalClinicalSummarizer asynchronously for the clinical chatbot.
    Replaces synchronous inline LLM calls in chatbot_interface.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    summarizer = UniversalClinicalSummarizer(gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key)

    conversation_context = conversation_context or []

    logger.info(f"[Celery Task] Processing chatbot input ({len(combined_input)} chars)")

    try:
        with AITimer() as timer:
            summary_html = summarizer.answer(combined_input, conversation_history=conversation_context)

        ai_mode = AIMode.LIVE_LLM if (gemini_api_key or nvidia_api_key) else AIMode.OFFLINE_FALLBACK

        raw_text_response = bleach.clean(summary_html, tags=[], strip=True)
        raw_text_response = re.sub(r'Response generated on.*', '', raw_text_response, flags=re.DOTALL)
        raw_text_response = re.sub(r'Powered by Gemini AI.*', '', raw_text_response, flags=re.DOTALL)
        raw_text_response = re.sub(r'\s{2,}', ' ', raw_text_response).strip()

        log_ai_call(
            feature='clinical_chatbot',
            mode=ai_mode,
            input_summary=combined_input[:200],
            output_summary=raw_text_response[:200],
            latency_ms=timer.elapsed_ms
        )

        return {
            'status': 'SUCCESS',
            'summary_html': summary_html,
            'raw_text': raw_text_response,
            'patient_id': patient_id,
            'input_note': combined_input,
        }
    except Exception as e:
        logger.error(f"[Celery Task {task_id}] Error generating response: {e}", exc_info=True)
        log_ai_call(
            feature='clinical_chatbot',
            mode=AIMode.OFFLINE_FALLBACK,
            input_summary=(combined_input or '')[:200],
            output_summary='',
            error=str(e)
        )
        raise e
