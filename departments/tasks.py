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
    combined_input: str,
    conversation_context: list | None = None,
    patient_id: str | None = None,
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


@shared_task(name="departments.tasks.process_soap_note_ai_analysis")
def process_soap_note_ai_analysis(note_id: int):
    """
    Celery task: trigger the FastAPI NLP service for a saved SOAP note.

    Replaces the synchronous requests.post() that used to sit in the doctor's
    request path and block the consultation for up to 30 seconds whenever the
    NLP worker was slow or down.
    """
    import requests

    logger.info("[Celery] Triggering NLP analysis for SOAP note %s", note_id)
    try:
        response = requests.post(
            "http://127.0.0.1:8000/process_note",
            json={"note_id": note_id},
            timeout=30,
        )
        response.raise_for_status()
        logger.info("[Celery] NLP analysis completed for SOAP note %s", note_id)
        return {"status": "ok", "note_id": note_id}
    except requests.exceptions.RequestException as exc:
        logger.error(
            "[Celery] NLP analysis trigger failed for note %s: %s", note_id, exc
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
# Pharmacy Inventory Alerts (near-expiry + low stock)
# ---------------------------------------------------------------------------


@shared_task(name="departments.tasks.pharmacy_inventory_alerts")
def pharmacy_inventory_alerts():
    """
    Celery beat task: daily pharmacy stock scan for near-expiry batches and
    drugs below their reorder level. Logs a summary; wire to
    departments/notifications dispatch as the next integration step.

    Schedule: 04:00 UTC / 07:00 EAT (set in celery_app.py beat schedule).
    """
    from app import app

    logger.info("[Celery] Pharmacy inventory alert scan starting …")
    try:
        with app.app_context():
            from departments.pharmacy.fefo import check_pharmacy_inventory_alerts

            alerts = check_pharmacy_inventory_alerts(near_expiry_days=60)
            logger.info(
                "[Celery] Pharmacy alerts: %d near-expiry batches, %d low-stock drugs",
                alerts["expiry_count"],
                alerts["reorder_count"],
            )
            return {
                "status": "ok",
                "expiry_alerts": alerts["expiry_count"],
                "reorder_alerts": alerts["reorder_count"],
            }
    except Exception as exc:
        logger.error("[Celery] Pharmacy inventory alert scan failed: %s", exc)
        raise


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
    except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
        logger.error("[Celery] LOINC sync failed: %s", exc)
        raise
