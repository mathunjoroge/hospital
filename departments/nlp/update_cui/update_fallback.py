# update_fallback.py
from apscheduler.schedulers.background import BackgroundScheduler
from departments.nlp.nlp_common import update_fallback_cui_map
from departments.nlp.logging_setup import get_logger

logger = get_logger(__name__)

def schedule_fallback_update():
    """Schedule periodic updates of FALLBACK_CUI_MAP."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        update_fallback_cui_map,
        'cron',
        hour=0,  # Run daily at midnight
        id='update_fallback_cui_map'
    )
    logger.info("Scheduled FALLBACK_CUI_MAP update at midnight daily")
    scheduler.start()

if __name__ == "__main__":
    schedule_fallback_update()
    try:
        import time
        while True:
            time.sleep(3600)  # Keep script running
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler")