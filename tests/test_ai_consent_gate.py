"""
tests/test_ai_consent_gate.py
──────────────────────────────
Tests for Gap 4: AI Consent Gate on Patient-Scoped Routes & AI features.
Verifies DPA 2019 compliance requirement for AI processing consent.
"""

import unittest
from datetime import date
from unittest.mock import patch

from werkzeug.security import generate_password_hash

from app import app
from departments.models.compliance import grant_patient_consent, has_ai_consent
from departments.models.records import Patient
from departments.models.user import User
from extensions import db, limiter


class AIConsentGateTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['RATELIMIT_ENABLED'] = False
        limiter.enabled = False
        if hasattr(limiter, '_storage') and hasattr(limiter._storage, 'reset'):
            limiter._storage.reset()
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()

        # Seed admin user
        admin = User(
            username='test_admin_ai',
            password=generate_password_hash('password123', method='pbkdf2:sha256'),
            role='admin'
        )
        db.session.add(admin)

        # Create test patient with all required fields
        self.patient = Patient(
            patient_id="P-TEST-CONSENT",
            name="Consent Test Patient",
            place_of_residence="Nairobi",
            sex="Female",
            date_of_birth=date(1990, 1, 1),
            marital_status="Single",
            contact="0712345678",
            next_of_kin="Kin Test",
            relationship_with_next_of_kin="Sibling",
            next_of_kin_contact="0787654321",
            emergency_contact="0711111111"
        )
        db.session.add(self.patient)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def _login(self):
        return self.app.post('/login', data={'username': 'test_admin_ai', 'password': 'password123'})

    def test_has_ai_consent_helper(self):
        """Test has_ai_consent returns True only when explicit ai_diagnosis consent is granted and unrevoked."""
        # 1. No consent record exists
        self.assertFalse(has_ai_consent("P-TEST-CONSENT"))

        # 2. Grant consent
        consent = grant_patient_consent("P-TEST-CONSENT", "ai_diagnosis")
        self.assertTrue(has_ai_consent("P-TEST-CONSENT"))

        # 3. Revoke consent
        consent.revoke()
        db.session.commit()
        self.assertFalse(has_ai_consent("P-TEST-CONSENT"))

    def test_chatbot_refuses_when_consent_missing(self):
        """POST to /medicine/chatbot with patient_id when consent is missing returns 403 refusal."""
        self._login()

        response = self.app.post(
            '/medicine/chatbot',
            data={
                'clinical_note': 'Patient presenting with fever',
                'patient_id': 'P-TEST-CONSENT'
            }
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn(b"AI-assisted summary unavailable", response.data)
        self.assertIn(b"patient has not consented", response.data)

    def test_chatbot_allows_when_consent_granted(self):
        """POST to /medicine/chatbot with patient_id when consent is granted permits AI processing."""
        grant_patient_consent("P-TEST-CONSENT", "ai_diagnosis")
        self._login()

        with patch('departments.tasks.process_clinical_chatbot_task.delay') as mock_delay:
            mock_delay.return_value.id = "mock-task-1"
            response = self.app.post(
                '/medicine/chatbot',
                data={
                    'clinical_note': 'Patient presenting with fever',
                    'patient_id': 'P-TEST-CONSENT'
                }
            )
            self.assertEqual(response.status_code, 202)
            mock_delay.assert_called_once()

    def test_chatbot_allows_ungated_query_without_patient_id(self):
        """POST to /medicine/chatbot without patient_id proceeds ungated."""
        self._login()

        with patch('departments.tasks.process_clinical_chatbot_task.delay') as mock_delay:
            mock_delay.return_value.id = "mock-task-2"
            response = self.app.post(
                '/medicine/chatbot',
                data={
                    'clinical_note': 'General clinical guidelines query'
                }
            )
            self.assertEqual(response.status_code, 202)
            mock_delay.assert_called_once()

    def test_oncology_summary_refuses_when_consent_missing(self):
        """GET/POST to /medicine/oncology/ai_summary/<patient_id> without consent returns 403."""
        self._login()
        response = self.app.post('/medicine/oncology/ai_summary/P-TEST-CONSENT')
        self.assertEqual(response.status_code, 403)
        self.assertIn(b"AI_CONSENT_REQUIRED", response.data)

    def test_oncology_summary_allows_when_consent_granted(self):
        """GET/POST to /medicine/oncology/ai_summary/<patient_id> with consent returns 200."""
        grant_patient_consent("P-TEST-CONSENT", "ai_diagnosis")
        self._login()
        with patch('departments.nlp.chatbot.UniversalClinicalSummarizer.answer') as mock_answer:
            mock_answer.return_value = "Oncology clinical summary response"
            response = self.app.post('/medicine/oncology/ai_summary/P-TEST-CONSENT')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Oncology clinical summary response", response.data)

    def test_ai_discovery_refuses_when_consent_missing(self):
        """POST to /pharmacy/ai_discovery with patient_id when consent missing returns 403."""
        self._login()
        response = self.app.post(
            '/pharmacy/ai_discovery',
            data={'action': 'molmim', 'patient_id': 'P-TEST-CONSENT', 'target_properties': 'logp < 3'}
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn(b"AI_CONSENT_REQUIRED", response.data)

    def test_ai_discovery_allows_when_consent_granted(self):
        """POST to /pharmacy/ai_discovery with patient_id when consent granted permits execution."""
        grant_patient_consent("P-TEST-CONSENT", "ai_diagnosis")
        self._login()
        with patch('departments.nlp.src.nvidia_client.NvidiaNIMClient.generate_molecules') as mock_gen:
            mock_gen.return_value = []
            response = self.app.post(
                '/pharmacy/ai_discovery',
                data={'action': 'molmim', 'patient_id': 'P-TEST-CONSENT', 'target_properties': 'logp < 3'}
            )
            self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()

