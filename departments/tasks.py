import logging
import os
import re

import bleach

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
from departments.billing.ward_charges import post_daily_ward_charges
from departments.nlp.chatbot import UniversalClinicalSummarizer

logger = logging.getLogger(__name__)


@shared_task
def process_clinical_chatbot_task(
    combined_input: str, conversation_context: list | None = None, patient_id: str | None = None
):
    """
    Celery task to run UniversalClinicalSummarizer asynchronously for the clinical chatbot.
    Replaces synchronous inline LLM calls in chatbot_interface.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    summarizer = UniversalClinicalSummarizer(
        gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key
    )

    conversation_context = conversation_context or []

    logger.info(f"[Celery Task] Processing chatbot input ({len(combined_input)} chars)")

    try:
        with AITimer() as timer:
            summary_html = summarizer.answer(
                combined_input, conversation_history=conversation_context
            )

        ai_mode = (
            AIMode.LIVE_LLM
            if (gemini_api_key or nvidia_api_key)
            else AIMode.OFFLINE_FALLBACK
        )

        raw_text_response = bleach.clean(summary_html, tags=[], strip=True)
        raw_text_response = re.sub(
            r"Response generated on.*", "", raw_text_response, flags=re.DOTALL
        )
        raw_text_response = re.sub(
            r"Powered by Gemini AI.*", "", raw_text_response, flags=re.DOTALL
        )
        raw_text_response = re.sub(r"\s{2,}", " ", raw_text_response).strip()

        log_ai_call(
            feature="clinical_chatbot",
            mode=ai_mode,
            input_summary=combined_input[:200],
            output_summary=raw_text_response[:200],
            latency_ms=timer.elapsed_ms,
        )

        return {
            "status": "SUCCESS",
            "summary_html": summary_html,
            "raw_text": raw_text_response,
            "patient_id": patient_id,
            "input_note": combined_input,
        }
    except Exception as e:
        logger.exception("[Celery Task] Error generating response: ")
        log_ai_call(
            feature="clinical_chatbot",
            mode=AIMode.OFFLINE_FALLBACK,
            input_summary=(combined_input or "")[:200],
            output_summary="",
            error=str(e),
        )
        raise


# --- T3.5: Ward Daily Charges ---

def scheduled_midnight_ward_charges():
    """Triggered by cron/celery at midnight."""
    try:
        post_daily_ward_charges()
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error posting ward charges: {e}")
# --------------------------------


# ---------------------------------------------------------------------------
# ICD-10 Nightly Re-Sync (DECISIONS_PENDING #4 resolved 2026-09-13)
# ---------------------------------------------------------------------------

@shared_task(name="departments.tasks.sync_icd10_codes")
def sync_icd10_codes():
    """
    Celery beat task: nightly re-sync of WHO ICD-10 codes into the local DB.

    Schedule: 00:00 UTC / 03:00 EAT (set in celery_app.py beat schedule).

    Uses import_from_who_api() which upserts — it only writes rows that are
    new or changed, so the DB footprint stays clean even over many runs.
    """
    from app import app

    logger.info("[Celery] ICD-10 nightly sync starting …")
    try:
        with app.app_context():
            from departments.medicine.icd10_importer import import_from_who_api
            count = import_from_who_api()
            logger.info("[Celery] ICD-10 sync complete: %d codes upserted.", count)
            return {"status": "ok", "codes_upserted": count}
    except Exception as exc:  # noqa: BLE001
        logger.error("[Celery] ICD-10 sync failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# SNOMED CT Nightly Sync (DECISIONS_PENDING #22 resolved 2026-09-13)
# ---------------------------------------------------------------------------

@shared_task(name="departments.tasks.sync_snomed_codes")
def sync_snomed_codes():
    """
    Celery beat task: nightly sync of SNOMED CT codes from UMLS API.
    Schedule: 00:30 UTC / 03:30 EAT (set in celery_app.py beat schedule).
    """
    from app import app

    logger.info("[Celery] SNOMED CT nightly sync starting …")
    try:
        with app.app_context():
            from departments.medicine.snomed_importer import import_from_umls_api
            count = import_from_umls_api(fetch_live_api=True)
            logger.info("[Celery] SNOMED CT sync complete: %d codes upserted.", count)
            return {"status": "ok", "codes_upserted": count}
    except Exception as exc:  # noqa: BLE001
        logger.error("[Celery] SNOMED CT sync failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# LOINC Nightly Sync (DECISIONS_PENDING #25 resolved 2026-09-13)
# ---------------------------------------------------------------------------

@shared_task(name="departments.tasks.sync_loinc_codes")
def sync_loinc_codes():
    """
    Celery beat task: nightly sync of LOINC codes from UMLS API.
    Schedule: 01:00 UTC / 04:00 EAT (set in celery_app.py beat schedule).
    """
    from app import app

    logger.info("[Celery] LOINC nightly sync starting …")
    try:
        with app.app_context():
            from departments.medicine.loinc_importer import import_from_umls_api
            count = import_from_umls_api(fetch_live_api=True)
            logger.info("[Celery] LOINC sync complete: %d codes upserted.", count)
            return {"status": "ok", "codes_upserted": count}
    except Exception as exc:  # noqa: BLE001
        logger.error("[Celery] LOINC sync failed: %s", exc)
        raise

