"""
tests/test_flow_audit_retest.py
Verification suite for patient flow audit fixes across:
1. HL7 lab ingestion stage advancement
2. Interleaved lab & imaging completion orderings
3. M-Pesa payment callback settlement & stage advancement
4. RCM claim approval settlement & stage advancement
5. Ward / Admitted queue mapping
6. Stale / abandoned encounter automated cleanup
"""
from datetime import datetime, timedelta, timezone

import pytest

from departments.api.hl7_receiver import ingest_lab_result
from departments.billing.mpesa_engine import MpesaService, MpesaTransaction
from departments.models.billing import Billing, Charge, ChargeCategory
from departments.models.encounter import Encounter
from departments.models.medicine import Imaging, RequestedImage, RequestedLab
from departments.models.records import Patient
from departments.rcm.claims_scrubber_engine import ClaimsScrubberEngine
from departments.rcm.models import ClaimSubmission
from departments.shared.queue_service import queue_for
from departments.shared.visit_closure import cleanup_stale_encounters
from extensions import db


@pytest.fixture
def test_patient(app):
    with app.app_context():
        p = Patient.query.filter_by(patient_id="P_AUDIT_01").first()
        if not p:
            p = Patient(
                patient_id="P_AUDIT_01",
                name="Audit Test Patient",
                dob=datetime(1990, 1, 1).date(),
                gender="Female",
            )
            db.session.add(p)
            db.session.commit()
        yield p


@pytest.fixture
def test_charge(app):
    with app.app_context():
        cat = ChargeCategory.query.first()
        if not cat:
            cat = ChargeCategory(name="Consultation")
            db.session.add(cat)
            db.session.commit()
        ch = Charge.query.first()
        if not ch:
            ch = Charge(name="Consultation Fee", category_id=cat.id, cost=500.0)
            db.session.add(ch)
            db.session.commit()
        yield ch


@pytest.fixture
def test_imaging_type(app):
    with app.app_context():
        img_type = Imaging.query.first()
        if not img_type:
            img_type = Imaging(imaging_type="X-Ray Chest", cost=1000.0)
            db.session.add(img_type)
            db.session.commit()
        yield img_type


def test_hl7_ingest_advances_stage(app, test_patient):
    with app.app_context():
        enc = Encounter(patient_id=test_patient.patient_id, stage="AWAITING_LAB", status="ACTIVE")
        db.session.add(enc)
        db.session.commit()

        req_lab = RequestedLab(patient_id=test_patient.patient_id, encounter_id=enc.id, lab_test_id=1, status=0)
        db.session.add(req_lab)
        db.session.commit()

        # Ingest HL7 lab result
        ingest_lab_result(
            patient_id=test_patient.patient_id,
            parameter_name="Hemoglobin",
            result_value=13.5,
            unit="g/dL",
            lab_test_id=1,
        )

        db.session.refresh(req_lab)
        db.session.refresh(enc)

        assert req_lab.status == 1
        assert enc.stage == "WAITING_DOCTOR_RESULTS"


def test_interleaved_completion_orderings(app, test_patient, test_imaging_type):
    with app.app_context():
        enc = Encounter(patient_id=test_patient.patient_id, stage="AWAITING_LAB", status="ACTIVE")
        db.session.add(enc)
        db.session.commit()

        lab = RequestedLab(patient_id=test_patient.patient_id, encounter_id=enc.id, lab_test_id=2, status=0)
        img = RequestedImage(patient_id=test_patient.patient_id, encounter_id=enc.id, imaging_id=test_imaging_type.id, status=0)
        db.session.add_all([lab, img])
        db.session.commit()

        # Lab finishes first -> moves to AWAITING_IMAGING
        lab.status = 1
        db.session.commit()
        from departments.shared.visit_closure import advance_after_completion
        advance_after_completion(test_patient.patient_id)

        db.session.refresh(enc)
        assert enc.stage == "AWAITING_IMAGING"

        # Imaging finishes second -> moves to WAITING_DOCTOR_RESULTS
        img.status = 1
        db.session.commit()
        advance_after_completion(test_patient.patient_id)

        db.session.refresh(enc)
        assert enc.stage == "WAITING_DOCTOR_RESULTS"


def test_mpesa_callback_settles_bills_and_advances_stage(app, test_patient, test_charge):
    with app.app_context():
        enc = Encounter(patient_id=test_patient.patient_id, stage="AWAITING_FINAL_BILLING", status="ACTIVE")
        db.session.add(enc)
        db.session.commit()

        bill = Billing(patient_id=test_patient.patient_id, charge_id=test_charge.id, quantity=1, total_cost=500.0, status=0)
        tx = MpesaTransaction(
            phone_number="254712345678",
            amount=500.0,
            reference=test_patient.patient_id,
            checkout_request_id="CHK_STK_AUDIT_01",
            status="PENDING",
        )
        db.session.add_all([bill, tx])
        db.session.commit()

        # Simulate Safaricom STK Push success callback
        payload = {
            "Body": {
                "stkCallback": {
                    "CheckoutRequestID": "CHK_STK_AUDIT_01",
                    "ResultCode": 0,
                    "ResultDesc": "The service request has been processed successfully.",
                    "CallbackMetadata": {
                        "Item": [{"Name": "Mpesa Receipt Number", "Value": "NL12345678"}]
                    },
                }
            }
        }
        success = MpesaService().process_callback(payload)
        assert success is True

        db.session.refresh(bill)
        db.session.refresh(enc)

        assert bill.status == 1
        assert enc.stage in ("DISCHARGED", None) or enc.status == "DISCHARGED"


def test_rcm_claim_approval_settles_bills_and_advances_stage(app, test_patient, test_charge):
    with app.app_context():
        enc = Encounter(patient_id=test_patient.patient_id, stage="AWAITING_FINAL_BILLING", status="ACTIVE")
        db.session.add(enc)
        db.session.commit()

        bill = Billing(patient_id=test_patient.patient_id, charge_id=test_charge.id, quantity=1, total_cost=1200.0, status=0)
        today = datetime.now(timezone.utc).date()
        claim = ClaimSubmission(
            id="CLM_AUDIT_835",
            patient_id=test_patient.patient_id,
            billed_amount=1200.0,
            status="SUBMITTED",
            service_start_date=today,
            service_end_date=today,
        )
        db.session.add_all([bill, claim])
        db.session.commit()

        # Parse X12 835 ERA remittance with paid claim status
        edi_content = (
            "ISA*00*          *00*          *ZZ*SHA             *ZZ*HOSPITAL        *260915*1000*U*00401*000000001*0*P*:~\n"
            "BPR*I*1200.00*C*ACH*CTX*01*999999999*DA*123456*1999999999*123456789*20260915~\n"
            "CLP*CLM_AUDIT_835*1*1200.00*1200.00*SHA_REF_01*11~\n"
            "SE*4*0001~"
        )
        res = ClaimsScrubberEngine.parse_and_apply_edi_835(edi_content)
        assert res["total_claims_processed"] == 1

        db.session.refresh(bill)
        db.session.refresh(enc)

        assert bill.status == 1
        assert enc.stage in ("DISCHARGED", None) or enc.status == "DISCHARGED"


def test_ward_queue_surfaces_admitted_encounters(app, test_patient):
    with app.app_context():
        enc = Encounter(patient_id=test_patient.patient_id, stage="ADMITTED", status="ACTIVE")
        db.session.add(enc)
        db.session.commit()

        ward_queue = queue_for("ward")
        patient_ids = [e.patient_id for e in ward_queue]
        assert test_patient.patient_id in patient_ids


def test_stale_encounter_cleanup(app, test_patient):
    with app.app_context():
        stale_time = datetime.now(timezone.utc) - timedelta(hours=30)
        enc = Encounter(patient_id=test_patient.patient_id, stage="REGISTERED_UNPAID", status="ACTIVE", started_at=stale_time)
        db.session.add(enc)
        db.session.commit()

        cancelled_count = cleanup_stale_encounters(max_hours=24)
        assert cancelled_count >= 1

        db.session.refresh(enc)
        assert enc.status == "CANCELLED"
        assert enc.stage == "CANCELLED"
