import sys
import os
import unittest

sys.path.insert(0, '/home/mathu/projects/hospital')

from departments.nlp.src.nvidia_client import NvidiaNIMClient, CANCER_TYPES, AMR_IPC_CATEGORIES
from departments.nlp.summarizer import ClinicalSummarizer
from departments.nlp.src.nlp import DiseasePredictor

class TestNvidiaNIMClient(unittest.TestCase):

    def setUp(self):
        self.client = NvidiaNIMClient()

    def test_client_initialization(self):
        """Test NvidiaNIMClient initializes cleanly."""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.model, "meta/llama-3.2-11b-vision-instruct")

    def test_cancer_risk_prediction_fallback(self):
        """Test offline rule-based fallback for cancer risk prediction."""
        note_text = "Patient presents with a persistent lump in breast, nipple discharge, and abnormal mammogram."
        res = self.client.predict_cancer_risk(note_text)
        
        self.assertIsInstance(res, dict)
        self.assertIn("breast cancer", res)
        # Breast cancer probability should be highest among all cancer types
        max_cancer = max(res, key=res.get)
        self.assertEqual(max_cancer, "breast cancer")

    def test_amr_ipc_prediction_high_risk(self):
        """Test AMR/IPC prediction for high resistance keywords."""
        note_text = "Patient diagnosed with MRSA bacteremia. Staff noted unwashed hands and PPE breach."
        res = self.client.predict_amr_ipc(note_text)

        self.assertIsInstance(res, dict)
        self.assertGreater(res["amr_high"], 0.5)
        self.assertGreater(res["ipc_inadequate"], 0.5)

    def test_amr_ipc_prediction_adequate_ipc(self):
        """Test AMR/IPC prediction with proper isolation protocol."""
        note_text = "Patient placed in strict isolation with PPE, glove, mask, and hand hygiene observed."
        res = self.client.predict_amr_ipc(note_text)

        self.assertIsInstance(res, dict)
        self.assertGreater(res["ipc_adequate"], 0.5)

    def test_summarize_note(self):
        """Test clinical note summarization."""
        note_text = "A 55-year-old male presents with severe chest pain radiating to the left arm. History of hypertension. Assessment: Acute Coronary Syndrome. Plan: ECG, troponin, aspirin."
        summary = self.client.summarize_note(note_text)

        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

    def test_analyze_radiology(self):
        """Test DICOM radiology AI analysis."""
        res = self.client.analyze_radiology(
            modality="CT",
            body_part="Chest",
            description="Rule out pulmonary embolism",
            symptoms="Acute dyspnea and pleuritic chest pain"
        )

        self.assertIsInstance(res, dict)
        self.assertEqual(res["status"], "success")
        self.assertIn("predictions", res)
        self.assertIn("confidence", res)
        self.assertIn("impression", res)

class TestClinicalSummarizer(unittest.TestCase):

    def test_summarize_dict_input(self):
        summarizer = ClinicalSummarizer()
        sample_dict = {
            "hpi": "60yo female with cough and fever.",
            "assessment": "Community acquired pneumonia.",
            "recommendation": "Azithromycin 500mg daily."
        }
        res = summarizer.summarize(sample_dict)
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)

if __name__ == "__main__":
    unittest.main()
