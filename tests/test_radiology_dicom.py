"""
tests/test_radiology_dicom.py
──────────────────────────────
Unit tests for Task 3.5: Radiology & DICOM Integration
"""

from datetime import datetime

import pytest

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.imaging import ImagingResult
from departments.models.medicine import Imaging, RequestedImage
from departments.models.records import Patient


@pytest.fixture
def radiology_app(app):
    return app


@pytest.fixture
def radiology_client(radiology_app):
    return radiology_app.test_client()


@pytest.fixture
def radiology_data(radiology_app):
    patient = Patient(
        patient_id="PT-RAD-01",
        name="Rad Test Patient",
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=datetime(1985, 5, 5).date(),
        marital_status="Married",
        contact="0700112233",
        next_of_kin="Kin",
        relationship_with_next_of_kin="Spouse",
        next_of_kin_contact="0700112233",
        emergency_contact="0700112233",
    )
    db.session.add(patient)

    imaging = Imaging(imaging_type="X-Ray Chest", cost=1500.0)
    db.session.add(imaging)
    db.session.commit()

    req_image = RequestedImage(
        patient_id=patient.patient_id,
        imaging_id=imaging.id,
        status=0,
        description="Routine chest x-ray",
    )
    db.session.add(req_image)
    db.session.commit()

    return {"patient": patient, "imaging": imaging, "req_image": req_image}


class TestDICOMIntegration:
    def test_modality_worklist(self, radiology_client, radiology_data):
        resp = radiology_client.get("/imaging/dicom/mwl")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "mwl" in data
        assert len(data["mwl"]) >= 1

        mwl_item = next(i for i in data["mwl"] if i["PatientID"] == "PT-RAD-01")
        assert mwl_item["PatientName"] == "Rad Test Patient"
        assert mwl_item["AccessionNumber"] == f"ACC-{radiology_data['req_image'].id}"

    def test_viewer_attachment_no_files(self, radiology_client, radiology_data):
        # Create an imaging result with no files
        req = radiology_data["req_image"]
        result_id = "RES-RAD-001"

        result = ImagingResult(
            result_id=result_id,
            patient_id=req.patient_id,
            imaging_id=req.imaging_id,
            dicom_file_path="",
            ai_generated=False,
        )
        db.session.add(result)
        req.result_id = result_id
        db.session.commit()

        resp = radiology_client.get(f"/imaging/dicom/viewer/{result_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 0
        assert "dicom_attachments" in data

    def test_structured_report(self, radiology_client, radiology_data):
        result_id = "RES-RAD-002"
        req = radiology_data["req_image"]

        result = ImagingResult(
            result_id=result_id,
            patient_id=req.patient_id,
            imaging_id=req.imaging_id,
            ai_generated=False,
        )
        db.session.add(result)
        req.result_id = result_id
        db.session.commit()

        resp = radiology_client.post(
            "/imaging/dicom/report",
            json={
                "result_id": result_id,
                "findings": "Lungs are clear.",
                "impression": "Normal chest X-ray.",
                "radiologist_id": 5,
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

        # Verify in DB
        db_res = ImagingResult.query.filter_by(result_id=result_id).first()
        assert "**Findings:**\nLungs are clear." in db_res.result_notes
        assert "**Impression:**\nNormal chest X-ray." in db_res.result_notes

        db_req = RequestedImage.query.get(req.id)
        assert db_req.status == 1  # Processed
