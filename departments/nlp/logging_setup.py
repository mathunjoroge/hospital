import logging
import os
from logging.handlers import RotatingFileHandler

class ThirdPartyFilter(logging.Filter):
    """Filter to suppress third-party logs."""
    def filter(self, record):
        # Suppress pymongo and spacy logs below INFO
        if record.name.startswith(('pymongo', 'spacy')) and record.levelno < logging.INFO:
            return False
        return True

def setup_logging(log_dir: str = None, debug: bool = False) -> logging.Logger:
    """Set up logging with separate handlers for detailed and error logs."""
    # Use project root logs directory or environment variable
    log_dir = log_dir or os.getenv('LOG_DIR', '/home/mathu/projects/hospital/logs')
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Failed to create log directory {log_dir}: {e}")
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)

    clinical_log_path = os.path.join(log_dir, 'clinical_ai.log')
    error_log_path = os.path.join(log_dir, 'error_log.txt')

    # Named logger
    logger = logging.getLogger('clinical_ai')
    if logger.handlers:  # Prevent duplicate handlers
        return logger

    logger.setLevel(logging.DEBUG)

    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler for detailed logs
    try:
        clinical_file_handler = RotatingFileHandler(
            clinical_log_path, maxBytes=10*1024*1024, backupCount=5
        )
        clinical_file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        clinical_file_handler.setFormatter(detailed_formatter)
        clinical_file_handler.addFilter(ThirdPartyFilter())
        logger.addHandler(clinical_file_handler)
    except Exception as e:
        print(f"Failed to set up clinical log file {clinical_log_path}: {e}")

    # File handler for error logs
    try:
        error_file_handler = RotatingFileHandler(
            error_log_path, maxBytes=10*1024*1024, backupCount=5
        )
        error_file_handler.setLevel(logging.WARNING)
        error_file_handler.setFormatter(simple_formatter)
        error_file_handler.addFilter(ThirdPartyFilter())
        logger.addHandler(error_file_handler)
    except Exception as e:
        print(f"Failed to set up error log file {error_log_path}: {e}")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    console_handler.addFilter(ThirdPartyFilter())
    logger.addHandler(console_handler)

    # Suppress pymongo and spacy logs below INFO
    logging.getLogger('pymongo').setLevel(logging.INFO)
    logging.getLogger('spacy').setLevel(logging.INFO)

    logger.info("Logging configuration set up successfully")
    return logger

def get_logger(name=None):
    import logging
    return logging.getLogger(name)
