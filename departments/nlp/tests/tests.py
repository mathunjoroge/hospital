import unittest
import multiprocessing
import time
import requests
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api import app, start_server
from src.database import setup_test_database
from src.nlp import DiseasePredictor
from src.config import get_config
import nltk
import logging

logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

class TestNLPApi(unittest.TestCase):
    """Unit tests for the Clinical NLP API."""
    
    server_process = None

    @classmethod
    def setUpClass(cls):
        """Start the server and set up test database."""
        from src.database import get_sqlite_connection
        cls.test_db = setup_test_database()
        def mock_sqlite_connection():
            return cls.test_db
        globals()['get_sqlite_connection'] = mock_sqlite_connection
        
        nltk.download('wordnet', quiet=True)
        DiseasePredictor.initialize()
        
        cls.server_process = multiprocessing.Process(target=start_server)
        cls.server_process.start()
        time.sleep(2)
        
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        """Stop the server."""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.join()
            logger.info("Server process terminated")

    def test_server_is_running(self):
        """Test if the server is running."""
        try:
            response = requests.get(f"http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}/docs", timeout=5)
            self.assertEqual(response.status_code, 200)
        except requests.ConnectionError:
            self.fail("Server is not running")

    def test_predict_endpoint(self):
        """Test the /predict endpoint with valid input."""
        response = self.client.post("/predict", json={"text": "Patient has fever and cough"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("primary_diagnosis", data)
        self.assertIn("differential_diagnoses", data)
        self.assertIn("processing_time", data)
        if data["primary_diagnosis"]:
            self.assertEqual(data["primary_diagnosis"]["disease"], "pneumonia")

    def test_predict_empty_text(self):
        """Test the /predict endpoint with empty input."""
        response = self.client.post("/predict", json={"text": ""})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["primary_diagnosis"])
        self.assertEqual(data["differential_diagnoses"], [])

    @patch('src.database.get_sqlite_connection')
    def test_process_note_endpoint(self, mock_db):
        """Test the /process_note endpoint with mocked database."""
        mock_db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {
            'id': 1, 'patient_id': 'PAT001', 'symptoms': 'fever, cough', 'created_at': '2025-07-02T14:00:00'
        }
        response = self.client.post("/process_note", json={"note_id": 1})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        html_content = response.text
        self.assertIn("Clinical Note Analysis - Note ID 1", html_content)
        self.assertIn("PAT001", html_content)
        self.assertIn("pneumonia", html_content)
        self.assertIn("Antibiotics, oxygen therapy", html_content)

    def test_process_note_not_found(self):
        """Test the /process_note endpoint with invalid note_id."""
        response = self.client.post("/process_note", json={"note_id": 999})
        self.assertEqual(response.status_code, 404)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Note Not Found", response.text)

    def test_rate_limit(self):
        """Test rate limiting on the /predict endpoint."""
        for _ in range(11):  # Exceed rate limit
            response = self.client.post("/predict", json={"text": "test"})
        self.assertEqual(response.status_code, 429)