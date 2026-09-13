"""
tests/test_clinical_trials_econsent.py
────────────────────────────────────────
Unit & integration tests for Gap #12: Clinical Trial Protocol & e-Consent Management.
"""

from datetime import date

from departments.clinical_trials.trials_engine import ClinicalTrialsEngine
from departments.models.records import Patient
from extensions import db


def test_protocol_creation_and_screening(app):
    """Test Clinical Trial Protocol registration and automated patient eligibility screening."""
    with app.app_context():
        # Register protocol
        protocol = ClinicalTrialsEngine.create_protocol(
            protocol_number="CT-2026-ONC99",
            title="Phase III Immunotherapy Trial for Solid Tumors",
            sponsor="BioPharma Global",
            principal_investigator="Dr. Alice Smith",
            phase="Phase III",
            target_enrollment=50,
            inclusion_criteria=["Age >= 18", "Histologically confirmed solid tumor"],
            exclusion_criteria=["Active infection", "Pregnancy"],
            treatment_arms=["Arm A: Pembrolizumab", "Arm B: Standard Chemotherapy"],
        )
        assert protocol.protocol_number == "CT-2026-ONC99"
        assert len(protocol.treatment_arms) == 2

        # Create active patient
        pat = Patient(patient_id="PAT_CT_01", name="Trial Patient One", sex="Female", date_of_birth=date(1980, 2, 14))
        db.session.add(pat)
        db.session.commit()

        # Screen patient eligibility
        screen_res = ClinicalTrialsEngine.screen_patient_eligibility(protocol.id, "PAT_CT_01")
        assert screen_res["is_eligible"] is True
        assert screen_res["screening_status"] == "ELIGIBLE"


def test_econsent_and_randomization(app):
    """Test e-Consent digital signature hashing & participant randomization."""
    with app.app_context():
        protocol = ClinicalTrialsEngine.create_protocol(
            protocol_number="CT-2026-CARD01",
            title="Phase II Heart Failure Study",
            sponsor="CardioHealth Corp",
            principal_investigator="Dr. Bob Johnson",
            phase="Phase II",
            treatment_arms=["Arm 1: Novel Inotrope", "Arm 2: Placebo"],
        )
        pat = Patient(patient_id="PAT_CT_02", name="Trial Patient Two", sex="Male", date_of_birth=date(1975, 8, 20))
        db.session.add(pat)
        db.session.commit()

        screen_res = ClinicalTrialsEngine.screen_patient_eligibility(protocol.id, "PAT_CT_02")
        pid = screen_res["participant_id"]

        # Record e-Consent
        consent_res = ClinicalTrialsEngine.record_econsent(pid, witness_name="Nurse Joy")
        assert consent_res["consent_status"] == "SIGNED_ECONSENT"
        assert consent_res["digital_signature_hash"] is not None
        assert len(consent_res["digital_signature_hash"]) == 64  # SHA-256 hex length

        # Randomize participant
        rand_res = ClinicalTrialsEngine.randomize_participant(pid)
        assert rand_res["enrollment_status"] == "RANDOMIZED"
        assert rand_res["randomized_arm"] in ["Arm 1: Novel Inotrope", "Arm 2: Placebo"]
        assert rand_res["protocol_current_enrollment"] == 1


def test_adverse_event_sae_logging(app):
    """Test Adverse Event (AE) and Serious Adverse Event (SAE) grade 1-5 escalation logic."""
    with app.app_context():
        protocol = ClinicalTrialsEngine.create_protocol(
            protocol_number="CT-2026-NEURO01",
            title="Phase I Neuroprotective Agent Trial",
            sponsor="NeuroLab Ltd",
            principal_investigator="Dr. Carol White",
            phase="Phase I",
        )
        pat = Patient(patient_id="PAT_CT_03", name="Trial Patient Three", sex="Female", date_of_birth=date(1993, 11, 5))
        db.session.add(pat)
        db.session.commit()

        screen_res = ClinicalTrialsEngine.screen_patient_eligibility(protocol.id, "PAT_CT_03")
        pid = screen_res["participant_id"]

        # Log Mild Grade 1 AE
        ae1 = ClinicalTrialsEngine.log_adverse_event(
            protocol_id=protocol.id,
            participant_id=pid,
            event_term="Mild Headache",
            severity_grade=1,
            is_serious_ae=False,
        )
        assert ae1["severity_grade"] == 1
        assert ae1["is_serious_ae"] is False
        assert ae1["irb_escalation_triggered"] is False

        # Log Grade 4 SAE (Life-Threatening)
        ae4 = ClinicalTrialsEngine.log_adverse_event(
            protocol_id=protocol.id,
            participant_id=pid,
            event_term="Severe Anaphylaxis",
            severity_grade=4,
            is_serious_ae=True,
            causality_assessment="PROBABLE",
        )
        assert ae4["severity_grade"] == 4
        assert ae4["is_serious_ae"] is True
        assert ae4["irb_escalation_triggered"] is True


def test_clinical_trials_api_endpoints(client, app, admin_user):
    """Test HTTP API endpoints for Protocols, Screening, e-Consent, Randomization, and SAE Logging."""
    with app.app_context():
        pat = Patient(patient_id="PAT_CT_API", name="API Trial Patient", sex="Male", date_of_birth=date(1988, 6, 12))
        db.session.add(pat)
        db.session.commit()

    # 1. Test POST /clinical-trials/api/protocols
    resp_proto = client.post(
        "/clinical-trials/api/protocols",
        json={
            "protocol_number": "CT-2026-API01",
            "title": "API Test Protocol",
            "sponsor": "API Sponsor",
            "principal_investigator": "Dr. Tester",
            "phase": "Phase III",
        },
    )
    assert resp_proto.status_code == 201
    proto_id = resp_proto.get_json()["protocol_id"]

    # 2. Test POST /clinical-trials/api/screen
    resp_screen = client.post(
        "/clinical-trials/api/screen",
        json={"protocol_id": proto_id, "patient_id": "PAT_CT_API"},
    )
    assert resp_screen.status_code == 200
    part_id = resp_screen.get_json()["participant_id"]

    # 3. Test POST /clinical-trials/api/econsent
    resp_consent = client.post(
        "/clinical-trials/api/econsent",
        json={"participant_id": part_id, "witness_name": "Dr. Witness"},
    )
    assert resp_consent.status_code == 200
    assert resp_consent.get_json()["consent_status"] == "SIGNED_ECONSENT"

    # 4. Test POST /clinical-trials/api/randomize
    resp_rand = client.post(
        "/clinical-trials/api/randomize",
        json={"participant_id": part_id},
    )
    assert resp_rand.status_code == 200
    assert resp_rand.get_json()["enrollment_status"] == "RANDOMIZED"

    # 5. Test POST /clinical-trials/api/adverse-event
    resp_ae = client.post(
        "/clinical-trials/api/adverse-event",
        json={
            "protocol_id": proto_id,
            "participant_id": part_id,
            "event_term": "Grade 3 Neutropenia",
            "severity_grade": 3,
        },
    )
    assert resp_ae.status_code == 201

    # 6. Test GET /clinical-trials/console
    resp_ui = client.get("/clinical-trials/console")
    assert resp_ui.status_code == 200
    assert b"Clinical Trial Protocol &amp; e-Consent Registry" in resp_ui.data
