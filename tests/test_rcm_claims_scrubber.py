"""
tests/test_rcm_claims_scrubber.py
──────────────────────────────────
Unit & integration tests for Gap #11: RCM & Pre-Submission Claims Scrubbing Engine.
"""

from datetime import date

from departments.models.records import Patient
from departments.rcm.claims_scrubber_engine import ClaimsScrubberEngine
from departments.rcm.models import ClaimSubmission
from extensions import db


def test_pre_claim_scrubbing_rules(app):
    """Test pre-submission scrubbing rules and denial risk scoring."""
    with app.app_context():
        pat = Patient(
            patient_id="PAT_RCM_01",
            name="RCM Patient One",
            sex="Male",
            date_of_birth=date(1990, 5, 10),
        )
        db.session.add(pat)
        db.session.commit()

        # 1. Clean Claim (< 20,000 KES, valid ICD-10)
        claim_clean = ClaimSubmission(
            patient_id="PAT_RCM_01",
            billed_amount=8500.0,
            primary_diagnosis_icd10="J02.9",
            service_start_date=date(2026, 9, 10),
            service_end_date=date(2026, 9, 10),
        )
        db.session.add(claim_clean)
        db.session.commit()

        report_clean = ClaimsScrubberEngine.scrub_claim(claim_clean.id)
        assert report_clean["is_clean"] is True
        assert report_clean["scrubbing_status"] == "CLEAN"
        assert report_clean["denial_risk_score"] == 0.0

        # 2. Dirty Claim (High Billed Amount > 20k without PreAuth, missing ICD-10)
        claim_dirty = ClaimSubmission(
            patient_id="PAT_RCM_01",
            billed_amount=45000.0,
            primary_diagnosis_icd10=None,
            service_start_date=date(2026, 9, 10),
            service_end_date=date(2026, 9, 10),
        )
        db.session.add(claim_dirty)
        db.session.commit()

        report_dirty = ClaimsScrubberEngine.scrub_claim(claim_dirty.id)
        assert report_dirty["is_clean"] is False
        assert report_dirty["scrubbing_status"] == "HAS_ERRORS"
        assert report_dirty["denial_risk_score"] > 0.0
        assert report_dirty["error_count"] >= 2


def test_edi_837_generator(app):
    """Test X12 EDI 837P Professional Claim text stream generation."""
    with app.app_context():
        pat = Patient(
            patient_id="PAT_RCM_837",
            name="EDI Test Patient",
            sex="Female",
            date_of_birth=date(1985, 3, 15),
        )
        db.session.add(pat)
        db.session.commit()

        claim = ClaimSubmission(
            patient_id="PAT_RCM_837",
            billed_amount=12000.0,
            primary_diagnosis_icd10="B50.9",
            service_start_date=date(2026, 9, 5),
            service_end_date=date(2026, 9, 5),
            status="CLEAN",
        )
        db.session.add(claim)
        db.session.commit()

        edi_text = ClaimsScrubberEngine.generate_edi_837(claim.id)
        assert "ISA*" in edi_text
        assert "ST*837*" in edi_text
        assert f"CLM*{claim.id}" in edi_text
        assert "HI*BK:B509~" in edi_text
        assert "SE*" in edi_text

        # Verify claim status updated to SUBMITTED
        c_updated = db.session.get(ClaimSubmission, claim.id)
        assert c_updated.status == "SUBMITTED"


def test_edi_835_remittance_parser(app):
    """Test X12 EDI 835 Remittance Advice (ERA) parsing & payment auto-reconciliation."""
    with app.app_context():
        pat = Patient(
            patient_id="PAT_RCM_835",
            name="Remittance Patient",
            sex="Male",
            date_of_birth=date(1992, 11, 20),
        )
        db.session.add(pat)
        db.session.commit()

        claim1 = ClaimSubmission(
            patient_id="PAT_RCM_835",
            billed_amount=15000.0,
            primary_diagnosis_icd10="E11.9",
            service_start_date=date(2026, 9, 1),
            service_end_date=date(2026, 9, 1),
            status="SUBMITTED",
        )
        claim2 = ClaimSubmission(
            patient_id="PAT_RCM_835",
            billed_amount=5000.0,
            primary_diagnosis_icd10="I10",
            service_start_date=date(2026, 9, 2),
            service_end_date=date(2026, 9, 2),
            status="SUBMITTED",
        )
        db.session.add_all([claim1, claim2])
        db.session.commit()

        cid1, cid2 = claim1.id, claim2.id

        edi_835_sample = f"""
        ISA*00*          *00*          *ZZ*SHA_KENYA       *ZZ*HOSPITAL_MAIN   *20260914*1200*U*00401*000000002*0*P*:~
        GS*HP*SHA_KENYA*HOSPITAL_MAIN*20260914*1200*2*X*004010X091A1~
        ST*835*0001~
        BPR*I*15000.00*C*ACH*CTX*01*999999999*DA*123456*1111111111**20260914~
        CLP*{cid1}*1*15000.00*15000.00*SHA-REF-101*11~
        CLP*{cid2}*2*5000.00*0.00*SHA-REF-102*11~
        CAS*CO*45*5000.00~
        SE*7*0001~
        GE*1*2~
        IEA*1*000000002~
        """

        result = ClaimsScrubberEngine.parse_and_apply_edi_835(edi_835_sample)
        assert result["total_claims_processed"] == 2
        assert result["total_paid_amount"] == 15000.0

        # Verify claim 1 is PAID
        c1 = db.session.get(ClaimSubmission, cid1)
        assert c1.status == "PAID"
        assert float(c1.paid_amount) == 15000.0

        # Verify claim 2 is DENIED
        c2 = db.session.get(ClaimSubmission, cid2)
        assert c2.status == "DENIED"
        assert float(c2.paid_amount) == 0.0


def test_rcm_claims_api_endpoints(client, app, admin_user):
    """Test HTTP API endpoints for RCM Scrubber, EDI 837, EDI 835, and Claims Console."""
    with app.app_context():
        pat = Patient(
            patient_id="PAT_RCM_API",
            name="API RCM Patient",
            sex="Female",
            date_of_birth=date(1989, 7, 25),
        )
        db.session.add(pat)
        db.session.commit()

        claim = ClaimSubmission(
            patient_id="PAT_RCM_API",
            billed_amount=9500.0,
            primary_diagnosis_icd10="J18.9",
            service_start_date=date(2026, 9, 8),
            service_end_date=date(2026, 9, 8),
        )
        db.session.add(claim)
        db.session.commit()
        cid = claim.id

    # 1. Test POST /rcm/api/scrub/<cid>
    resp_scrub = client.post(f"/rcm/api/scrub/{cid}")
    assert resp_scrub.status_code == 200
    assert resp_scrub.get_json()["is_clean"] is True

    # 2. Test GET /rcm/api/edi837/<cid>
    resp_837 = client.get(f"/rcm/api/edi837/{cid}")
    assert resp_837.status_code == 200
    assert "ISA*" in resp_837.get_json()["edi_content"]

    # 3. Test GET /rcm/api/denial-risk/<cid>
    resp_risk = client.get(f"/rcm/api/denial-risk/{cid}")
    assert resp_risk.status_code == 200
    assert resp_risk.get_json()["denial_risk_score"] == 0.0

    # 4. Test GET /rcm/claims-console
    resp_ui = client.get("/rcm/claims-console")
    assert resp_ui.status_code == 200
    assert b"RCM Claims Scrubbing &amp; EDI 837/835 Console" in resp_ui.data
