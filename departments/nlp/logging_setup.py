import logging
import os
from logging import StreamHandler, FileHandler

def setup_logging():
    """Set up logging with separate handlers for detailed and error logs."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    clinical_log_path = os.path.join(log_dir, 'clinical_ai.log')
    error_log_path = os.path.join(log_dir, 'error_log.txt')

    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    clinical_file_handler = FileHandler(clinical_log_path)
    clinical_file_handler.setLevel(logging.DEBUG)
    clinical_file_handler.setFormatter(detailed_formatter)

    error_file_handler = FileHandler(error_log_path)
    error_file_handler.setLevel(logging.WARNING)
    error_file_handler.setFormatter(simple_formatter)

    console_handler = StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[clinical_file_handler, error_file_handler, console_handler]
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging configuration set up successfully")
    return logger

logger = setup_logging()