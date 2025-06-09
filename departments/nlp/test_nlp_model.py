import unittest
import logging
import spacy
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
# Relative import for nlp_pipeline.py in the same directory
from departments.nlp.nlp_pipeline import SciBERTWrapper, get_nlp, logger, STOP_TERMS, extract_clinical_phrases

# Configure logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
test_logger = logging.getLogger(__name__)

class TestNLPModelInstallation(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.sample_text = "Patient reports nausea and vomiting, fever, and jaundice in eyes."
        self.model_name = "en_core_sci_sm"
        test_logger.info("Setting up test environment")

    def test_model_loading(self):
        """Test if en_core_sci_sm model loads successfully."""
        try:
            nlp = spacy.load(self.model_name, disable=["lemmatizer"])
            test_logger.info(f"Successfully loaded {self.model_name}")
            self.assertIsInstance(nlp, spacy.language.Language, f"{self.model_name} did not load as a spaCy Language object")
        except OSError as e:
            test_logger.error(f"Failed to load {self.model_name}: {e}")
            self.fail(f"Model {self.model_name} is not installed or failed to load: {e}")

    def test_scibert_wrapper_initialization(self):
        """Test SciBERTWrapper initialization with en_core_sci_sm."""
        try:
            wrapper = SciBERTWrapper(model_name=self.model_name, disable_linker=True)
            test_logger.info("SciBERTWrapper initialized successfully")
            self.assertIsInstance(wrapper.nlp, spacy.language.Language, "SciBERTWrapper did not initialize a spaCy Language object")
            self.assertNotIn("entity_linker", wrapper.nlp.pipe_names, "entity_linker was not removed from pipeline")
        except Exception as e:
            test_logger.error(f"SciBERTWrapper initialization failed: {e}")
            self.fail(f"SciBERTWrapper failed to initialize with {self.model_name}: {e}")

    def test_scibert_wrapper_entity_extraction(self):
        """Test entity extraction using SciBERTWrapper."""
        wrapper = SciBERTWrapper(model_name=self.model_name, disable_linker=True)
        entities = wrapper.extract_entities(self.sample_text)
        test_logger.info(f"Extracted entities: {entities}")
        self.assertIsInstance(entities, list, "Entity extraction did not return a list")
        self.assertGreaterEqual(len(entities), 1, "No entities extracted from sample text")
        # Check for individual terms since model splits compound phrases
        entity_texts = [ent[0].lower() for ent in entities]
        expected_terms = ["nausea", "vomiting", "fever", "jaundice"]
        for term in expected_terms:
            self.assertTrue(
                any(term in entity_text for entity_text in entity_texts),
                f"Expected term '{term}' not found in extracted entities: {entity_texts}"
            )

    @patch('nlp_pipeline.MongoClient')
    @patch('nlp_pipeline.SimpleConnectionPool')
    @patch('nlp_pipeline.load_knowledge_base')
    def test_get_nlp_pipeline(self, mock_load_knowledge_base, mock_pool, mock_mongo):
        """Test get_nlp() pipeline initialization with mocked MongoDB, PostgreSQL, and knowledge base."""
        # Mock knowledge base to include hepatic category
        mock_load_knowledge_base.return_value = {
            "medical_stop_words": set(),
            "version": "1.0",
            "symptoms": {"gastrointestinal": [], "hepatic": [], "general": []}
        }

        # Mock MongoDB client
        mock_mongo_instance = MagicMock()
        mock_mongo_instance.admin.command.return_value = {"ok": 1}
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.index_information.return_value = {"symptom_1": {"unique": True}}
        mock_collection.find.return_value.batch_size.return_value = []
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_instance.__getitem__.return_value = mock_db
        mock_mongo.return_value = mock_mongo_instance

        # Mock PostgreSQL pool
        mock_pool_instance = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance

        try:
            nlp = get_nlp()
            test_logger.info(f"NLP pipeline initialized with components: {nlp.pipe_names}")
            self.assertIsInstance(nlp, spacy.language.Language, "get_nlp did not return a spaCy Language object")
            expected_components = ["sentencizer", "symptom_matcher", "medspacy_context"]
            for component in expected_components:
                self.assertIn(component, nlp.pipe_names, f"Pipeline missing component: {component}")
        except Exception as e:
            test_logger.error(f"get_nlp pipeline initialization failed: {e}")
            self.fail(f"get_nlp pipeline initialization failed: {e}")

    def test_extract_clinical_phrases(self):
        """Test extract_clinical_phrases with sample text."""
        phrases = extract_clinical_phrases(self.sample_text)
        test_logger.info(f"Extracted clinical phrases: {phrases}")
        self.assertIsInstance(phrases, list, "extract_clinical_phrases did not return a list")
        self.assertGreaterEqual(len(phrases), 1, "No clinical phrases extracted from sample text")
        # Check for individual terms since model splits compound phrases
        expected_phrases = ["nausea", "vomiting", "fever", "jaundice"]
        for phrase in expected_phrases:
            cleaned_phrase = phrase.lower()
            if cleaned_phrase not in STOP_TERMS:
                self.assertIn(
                    cleaned_phrase,
                    [p.lower() for p in phrases],
                    f"Expected phrase '{cleaned_phrase}' not found in extracted phrases: {phrases}"
                )

    def tearDown(self):
        """Clean up test environment."""
        test_logger.info("Tearing down test environment")

if __name__ == "__main__":
    unittest.main()