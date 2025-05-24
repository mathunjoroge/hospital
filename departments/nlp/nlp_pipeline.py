import medspacy
from spacy.language import Language
from departments.nlp.logging_setup import get_logger

logger = get_logger()
_nlp_instance = None

def get_nlp():
    """Initialize or return the medspacy pipeline with ConText and UMLS linker."""
    global _nlp_instance
    if _nlp_instance is None:
        try:
            # Load the base SpaCy model
            logger.debug("Loading SpaCy model 'en_core_sci_sm'")
            import spacy
            _nlp_instance = spacy.load("en_core_sci_sm", disable=["ner"])  # Disable NER to avoid conflicts
            logger.debug("Initial pipeline after load: %s", _nlp_instance.pipe_names)

            # Add MedSpaCy components explicitly
            logger.debug("Adding medspacy_pyrush")
            _nlp_instance.add_pipe("medspacy_pyrush")
            
            logger.debug("Adding medspacy_target_matcher")
            _nlp_instance.add_pipe("medspacy_target_matcher")

            # Ensure medspacy_context is only added once
            if "medspacy_context" not in _nlp_instance.pipe_names:
                logger.debug("Adding medspacy_context to pipeline")
                _nlp_instance.add_pipe("medspacy_context")
            else:
                logger.debug("medspacy_context already in pipeline, skipping addition")

            # Add scispacy UMLS linker
            logger.debug("Adding scispacy_linker with UMLS configuration")
            _nlp_instance.add_pipe("scispacy_linker", config={"linker_name": "umls"})
            logger.info("MedSpaCy pipeline initialized: %s", _nlp_instance.pipe_names)
        except Exception as e:
            logger.error("Failed to initialize MedSpaCy pipeline: %s", str(e))
            _nlp_instance = None
            raise
    return _nlp_instance
