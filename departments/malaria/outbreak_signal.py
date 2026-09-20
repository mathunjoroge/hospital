"""
departments/malaria/outbreak_signal.py
───────────────────────────────────────
P7-11 — Malaria outbreak signal.

A Celery task runs nightly (or can be triggered manually) and compares
the 7-day rolling case count against 2× the 4-week rolling average.
If the threshold is exceeded, a CRITICAL notification is sent to the
facility Medical Officer via the existing notification dispatcher.

Thresholds are intentionally conservative: the signal should be
over-sensitive rather than under-sensitive for a public-health trigger.
A facility epidemiologist or Medical Officer must confirm before
escalating externally to the County Health team.
"""

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _get_outbreak_counts(db_session) -> dict:
    """
    Return 7-day and 4-week case counts from the malaria_cases table.
    Returns a dict with keys: count_7d, count_4w_avg, threshold.
    """
    from departments.malaria.models import MalariaCase

    now = datetime.now(timezone.utc)
    window_7d_start = now - timedelta(days=7)
    window_4w_start = now - timedelta(weeks=4)

    count_7d = (
        db_session.query(MalariaCase)
        .filter(MalariaCase.created_at >= window_7d_start)
        .count()
    )

    # 4-week average is computed as total cases in 4 weeks / 4
    count_4w_total = (
        db_session.query(MalariaCase)
        .filter(MalariaCase.created_at >= window_4w_start)
        .count()
    )
    count_4w_avg = count_4w_total / 4.0

    # Outbreak threshold: 2× 4-week weekly average
    threshold = count_4w_avg * 2

    return {
        "count_7d": count_7d,
        "count_4w_avg": count_4w_avg,
        "threshold": threshold,
    }


def check_malaria_outbreak_signal(db_session=None) -> dict:
    """
    Evaluate the outbreak signal.  Returns a result dict:
        {
            "signal": True/False,
            "count_7d": int,
            "count_4w_avg": float,
            "threshold": float,
            "message": str,
        }

    When signal=True the caller is responsible for dispatching an alert;
    this function does not side-effect (easier to unit-test).
    """
    from extensions import db as _db

    session = db_session or _db.session
    counts = _get_outbreak_counts(session)

    signal = (
        counts["count_4w_avg"] > 0  # avoid triggering when baseline is zero
        and counts["count_7d"] >= counts["threshold"]
    )

    message = (
        f"MALARIA OUTBREAK SIGNAL: {counts['count_7d']} confirmed cases in the past 7 days "
        f"exceeds 2× the 4-week weekly average ({counts['count_4w_avg']:.1f} cases/week). "
        f"Threshold was {counts['threshold']:.1f}. Immediate review by Medical Officer required."
        if signal
        else (
            f"No outbreak signal. 7-day count: {counts['count_7d']}, "
            f"4-week weekly average: {counts['count_4w_avg']:.1f}, "
            f"threshold: {counts['threshold']:.1f}."
        )
    )

    return {
        "signal": signal,
        "count_7d": counts["count_7d"],
        "count_4w_avg": counts["count_4w_avg"],
        "threshold": counts["threshold"],
        "message": message,
    }


# ── Celery task ──────────────────────────────────────────────────────────────

def register_outbreak_task(celery_app):
    """
    Register the nightly malaria outbreak check as a Celery beat task.

    Call this from celery_app.py after the Celery instance is created:
        from departments.malaria.outbreak_signal import register_outbreak_task
        register_outbreak_task(celery)
    """

    @celery_app.task(name="malaria.check_outbreak_signal", bind=True)
    def _check_outbreak_signal_task(self):  # noqa: ANN001
        """Nightly malaria outbreak signal evaluation (P7-11)."""
        try:
            result = check_malaria_outbreak_signal()
            if result["signal"]:
                logger.critical("MALARIA OUTBREAK SIGNAL: %s", result["message"])
                _dispatch_outbreak_alert(result)
            else:
                logger.info("Malaria outbreak check: %s", result["message"])
            return result
        except Exception:  # noqa: BLE001
            logger.exception("Malaria outbreak check task failed")
            raise self.retry(countdown=300, max_retries=2)

    return _check_outbreak_signal_task


def _dispatch_outbreak_alert(result: dict) -> None:
    """
    Send outbreak alert to the facility Medical Officer via the notification dispatcher.
    Falls back to a logger.critical call if the dispatcher is unavailable.
    """
    try:
        from departments.notifications.dispatcher import dispatch_notification

        dispatch_notification(
            event_type="MALARIA_OUTBREAK_SIGNAL",
            roles=["medical_officer", "admin"],
            subject="⚠️ Malaria Outbreak Signal Triggered",
            body=result["message"],
            severity="CRITICAL",
        )
        logger.info("Outbreak alert dispatched to Medical Officer.")
    except Exception:  # noqa: BLE001
        # Notification dispatch failure must never suppress the log
        logger.exception(
            "Failed to dispatch outbreak alert via notification dispatcher — "
            "see logger.critical above for the signal details."
        )
