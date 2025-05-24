import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from departments.models.medicine import SOAPNote, Base
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.batch_processing import update_single_soap_note
from departments.nlp.kb_updater import KnowledgeBaseUpdater
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base
from departments.nlp.knowledge_base_init import initialize_knowledge_files
import os
import json

# Use in-memory SQLite for testing
engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

class TestClinicalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.analyzer = ClinicalAnalyzer()
        self.kb_updater = KnowledgeBaseUpdater()
        # Clear knowledge base directory for clean test
        self.kb_dir = os.path.join(os.path.dirname(__file__), '..', 'departments', 'nlp', 'knowledge_base')
        if os.path.exists(self.kb_dir):
            for f in os.listdir(self.kb_dir):
                os.remove(os.path.join(self.kb_dir, f))

    def tearDown(self):
        self.session.close()

    def test_extract_clinical_features(self):
        note = SOAPNote(
            situation="Patient reports back pain and obesity, denies chest pain.",
            hpi="Back pain for 2 weeks, worse with movement.",
            assessment="Mechanical low back pain, obese.",
            patient_id="12345",
            symptoms="back pain, obesity",
            medication_history="None"
        )
        self.session.add(note)
        self.session.commit()
        features = self.analyzer.extract_clinical_features(note, expected_symptoms=["back pain", "obesity", "chest pain"])
        self.assertTrue(any(s['description'] == 'back pain' for s in features['symptoms']))
        self.assertTrue(any(s['description'] == 'obesity' for s in features['symptoms']))
        self.assertFalse(any(s['description'] == 'chest pain' for s in features['symptoms']), "Chest pain should be negated")
        back_pain_symptom = next(s for s in features['symptoms'] if s['description'] == 'back pain')
        self.assertEqual(back_pain_symptom['umls_cui'], 'C0004604')
        self.assertEqual(back_pain_symptom['semantic_type'], 'Sign or Symptom')

    def test_generate_differential_dx(self):
        note = SOAPNote(
            situation="Patient reports back pain, denies chest pain.",
            hpi="Back pain for 2 weeks, worse with movement.",
            assessment="Mechanical low back pain.",
            patient_id="12345",
            symptoms="back pain",
            medication_history="None"
        )
        self.session.add(note)
        self.session.commit()
        features = self.analyzer.extract_clinical_features(note, expected_symptoms=["back pain"])
        differentials = self.analyzer.generate_differential_dx(features)
        self.assertTrue(any(dx == "Mechanical low back pain" for dx, _, _ in differentials))
        self.assertFalse(any(dx.lower().startswith("myocardial") for dx, _, _ in differentials), "Negated chest pain should exclude myocardial infarction")

    def test_batch_processing(self):
        note = SOAPNote(
            situation="Patient reports back pain, denies chest pain.",
            hpi="Back pain for 2 weeks, worse with movement.",
            assessment="Mechanical low back pain.",
            patient_id="12345",
            symptoms="back pain",
            ai_notes=None,
            ai_analysis=None,
            medication_history="None"
        )
        self.session.add(note)
        self.session.commit()
        update_single_soap_note(note.id, self.analyzer)
        self.session.refresh(note)
        self.assertIsNotNone(note.ai_notes, "AI notes should be generated")
        self.assertIsNotNone(note.ai_analysis, "AI analysis should be generated")
        self.assertIn("Back Pain (CUI: C0004604", note.ai_notes, "UMLS data should be included in AI notes")
        self.assertNotIn("Chest pain", note.ai_notes, "Negated chest pain should not appear in AI notes")

    def test_knowledge_base_update(self):
        symptom = "knee pain"
        context = "Patient reports knee pain, denies chest pain."
        category = self.kb_updater.infer_category(symptom, context)
        synonyms = self.kb_updater.generate_synonyms(symptom)
        result = self.kb_updater.update_knowledge_base(symptom, category, synonyms, context)
        self.assertTrue(result, "Knowledge base should be updated for new symptom")
        self.assertIn(symptom.lower(), self.kb_updater.medical_terms, "Symptom should be in medical_terms")
        self.assertEqual(self.kb_updater.synonyms.get(symptom.lower()), synonyms, "Synonyms should be stored")
        negated_symptom = "chest pain"
        result = self.kb_updater.update_knowledge_base(negated_symptom, "cardiovascular", [], context)
        self.assertFalse(result, "Negated symptom should not be added")

    def test_generate_ai_summary(self):
        note = SOAPNote(
            situation="Male, back pain and obesity, denies chest pain.",
            hpi="Back pain for 2 weeks, worse with movement.",
            assessment="Primary Assessment: Mechanical low back pain.",
            patient_id="12345",
            symptoms="back pain, obesity",
            medication_history="Ibuprofen"
        )
        self.session.add(note)
        self.session.commit()
        summary = generate_ai_summary(note)
        self.assertIn("Chief Complaint: back pain and obesity, denies chest pain", summary)
        self.assertIn("HPI: Back pain for 2 weeks, worse with movement", summary)
        self.assertIn("Medications: Ibuprofen", summary)
        self.assertIn("Primary Diagnosis: Mechanical low back pain", summary)
        self.assertIn("Extracted Symptoms:", summary)
        self.assertIn("back pain (Category: musculoskeletal, CUI: C0004604, Semantic Type: Sign or Symptom)", summary.lower())
        self.assertIn("obesity (Category: musculoskeletal, CUI: C0028754, Semantic Type: Disease or Syndrome)", summary.lower())
        self.assertNotIn("chest pain", summary.lower(), "Negated chest pain should not appear in symptoms")

    def test_knowledge_base_init_and_io(self):
        # Test initialization
        initialize_knowledge_files()
        medical_terms_path = os.path.join(self.kb_dir, "medical_terms.json")
        synonyms_path = os.path.join(self.kb_dir, "synonyms.json")
        clinical_pathways_path = os.path.join(self.kb_dir, "clinical_pathways.json")
        self.assertTrue(os.path.exists(medical_terms_path), "medical_terms.json should be created")
        self.assertTrue(os.path.exists(synonyms_path), "synonyms.json should be created")
        self.assertTrue(os.path.exists(clinical_pathways_path), "clinical_pathways.json should be created")

        # Test loading
        kb = load_knowledge_base()
        self.assertIn("medical_terms", kb)
        self.assertIn("synonyms", kb)
        self.assertIn("clinical_pathways", kb)
        self.assertTrue(any(t["term"] == "back pain" and t["umls_cui"] == "C0004604" for t in kb["medical_terms"]))
        self.assertIn("back pain", kb["synonyms"])
        self.assertIn("musculoskeletal", kb["clinical_pathways"])
        self.assertIn("back pain|lower back pain|backache", kb["clinical_pathways"]["musculoskeletal"])

        # Test saving
        kb["medical_terms"].append({
            "term": "new symptom",
            "category": "general",
            "umls_cui": None,
            "semantic_type": "Unknown"
        })
        result = save_knowledge_base(kb)
        self.assertTrue(result, "Knowledge base should save successfully")
        with open(medical_terms_path, 'r') as f:
            saved_data = json.load(f)
        self.assertTrue(any(t["term"] == "new symptom" for t in saved_data), "New symptom should be saved")

if __name__ == '__main__':
    unittest.main()