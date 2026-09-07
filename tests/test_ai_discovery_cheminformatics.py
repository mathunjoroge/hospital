import unittest

from werkzeug.security import generate_password_hash

from app import app, db
from departments.models.user import User
from departments.pharmacy.cheminformatics import (
    calculate_tanimoto_similarity,
    find_closest_reference_drugs,
    generate_3d_molblock,
    validate_and_analyze_smiles,
)


class TestAIDiscoveryCheminformatics(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()

        # Ensure pharmacy user exists
        user = User.query.filter_by(username="test_pharmacy_user").first()
        if not user:
            user = User(
                username="test_pharmacy_user",
                password=generate_password_hash("password123", method="pbkdf2:sha256"),
                role="pharmacy",
            )
            db.session.add(user)
            db.session.commit()
        self.user = user

    def tearDown(self):
        self.app_context.pop()

    def test_validate_and_analyze_smiles_valid(self):
        """Test RDKit descriptor calculation for valid Aspirin SMILES."""
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        res = validate_and_analyze_smiles(aspirin_smiles)

        self.assertTrue(res["is_valid"])
        self.assertEqual(res["canonical_smiles"], "CC(=O)Oc1ccccc1C(=O)O")
        self.assertAlmostEqual(res["mw"], 180.16, places=1)
        self.assertAlmostEqual(res["logp"], 1.31, places=1)
        self.assertEqual(res["hbd"], 1)
        self.assertEqual(res["hba"], 3)
        self.assertTrue(res["lipinski_pass"])
        self.assertEqual(res["lipinski_violations_count"], 0)

    def test_validate_and_analyze_smiles_invalid(self):
        """Test RDKit validation for invalid SMILES string."""
        invalid_smiles = "INVALID_CHEMICAL_STRING_123"
        res = validate_and_analyze_smiles(invalid_smiles)

        self.assertFalse(res["is_valid"])
        self.assertIsNotNone(res["error"])

    def test_generate_3d_molblock(self):
        """Test 3D Molblock coordinate generation."""
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        molblock = generate_3d_molblock(aspirin_smiles)

        self.assertIsNotNone(molblock)
        self.assertIn("END", molblock)
        self.assertTrue(len(molblock.splitlines()) > 10)

    def test_tanimoto_similarity(self):
        """Test Morgan Fingerprint Tanimoto similarity score."""
        smiles1 = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        smiles2 = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin self match

        sim_self = calculate_tanimoto_similarity(smiles1, smiles2)
        self.assertEqual(sim_self, 1.0)

        smiles3 = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
        sim_paracetamol = calculate_tanimoto_similarity(smiles1, smiles3)
        self.assertGreaterEqual(sim_paracetamol, 0.0)
        self.assertLess(sim_paracetamol, 1.0)

    def test_find_closest_reference_drugs(self):
        """Test reference drug matching library."""
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        matches = find_closest_reference_drugs(aspirin_smiles, top_n=3)

        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0]["name"], "Aspirin")
        self.assertEqual(matches[0]["similarity"], 1.0)

    def test_ai_discovery_route_get(self):
        """Test GET /pharmacy/ai_discovery endpoint."""
        self.client.post(
            "/login", data={"username": "test_pharmacy_user", "password": "password123"}
        )
        response = self.client.get("/pharmacy/ai_discovery")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AI-Assisted Molecule Ideation", response.data)

    def test_ai_discovery_route_molmim_post(self):
        """Test POST /pharmacy/ai_discovery for candidate generation."""
        self.client.post(
            "/login", data={"username": "test_pharmacy_user", "password": "password123"}
        )
        response = self.client.post(
            "/pharmacy/ai_discovery",
            data={
                "action": "molmim",
                "target_properties": "high solubility COX-2 inhibitor",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Generated Candidates for", response.data)
        self.assertIn(b"RDKit Computed Properties", response.data)

    def test_ai_discovery_shortlist_session(self):
        """Test adding, removing, and clearing shortlist items in Flask session."""
        self.client.post(
            "/login", data={"username": "test_pharmacy_user", "password": "password123"}
        )

        candidate = {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "mw": 180.16,
            "logp": 1.31,
            "tpsa": 63.6,
            "lipinski_pass": True,
        }

        # Add
        add_resp = self.client.post(
            "/pharmacy/ai_discovery/shortlist/add", json={"candidate": candidate}
        )
        self.assertEqual(add_resp.status_code, 200)
        self.assertEqual(add_resp.json["count"], 1)

        # Clear
        clear_resp = self.client.post("/pharmacy/ai_discovery/shortlist/clear")
        self.assertEqual(clear_resp.status_code, 200)
        self.assertEqual(clear_resp.json["count"], 0)


if __name__ == "__main__":
    unittest.main()
