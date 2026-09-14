"""
tests/test_dicomweb_pacs.py
────────────────────────────
Unit & Integration tests for Johns Hopkins–Grade DICOMweb / PACS SCU integration.
Tests:
  - QIDO-RS, WADO-RS, STOW-RS DICOMweb client operations
  - DICOMService STOW-RS auto-push integration
  - Embedded OHIF Web Viewer console route (/imaging/dicom/ohif/<result_id>)
  - DICOMweb REST proxy routes (/imaging/dicom/qido, /imaging/dicom/wado/<study_uid>)
  - FHIR R4 ImagingStudy resource serialization & endpoint (/api/fhir/R4/ImagingStudy)
"""

from datetime import date
from unittest.mock import MagicMock, patch

from departments.imaging.dicomweb_client import DICOMwebClient, dicomweb_client
from departments.models.imaging import ImagingResult
from departments.models.medicine import Imaging
from departments.models.records import Patient
from extensions import db


def _create_test_patient(patient_id: str, name: str, sex: str = "Male") -> Patient:
    """Helper to instantiate valid Patient model."""
    p = Patient.query.filter_by(patient_id=patient_id).first()
    if not p:
        p = Patient(
            patient_id=patient_id,
            name=name,
            place_of_residence="Nairobi",
            sex=sex,
            date_of_birth=date(1990, 1, 1),
            marital_status="Single",
            contact="0700000000",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Self",
            next_of_kin_contact="0700000000",
            emergency_contact="0700000000",
        )
        db.session.add(p)
        db.session.commit()
    return p


class TestDICOMwebClient:
    """Test DICOMweb REST Client (QIDO-RS, WADO-RS, STOW-RS)."""

    def test_qido_search_studies(self):
        client = DICOMwebClient()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [{"0020000D": {"Value": ["1.2.3.4"]}}]
            res = client.qido_search_studies(patient_id="PT001", modality="CT")
            assert len(res) == 1
            assert res[0]["0020000D"]["Value"][0] == "1.2.3.4"

    def test_wado_retrieve_metadata(self):
        client = DICOMwebClient()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [{"00080060": {"Value": ["DX"]}}]
            metadata = client.wado_retrieve_metadata("1.2.3.4")
            assert len(metadata) == 1
            assert metadata[0]["00080060"]["Value"][0] == "DX"

    def test_stow_store_instances_file_not_found(self):
        client = DICOMwebClient()
        try:
            client.stow_store_instances("/nonexistent/file.dcm")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            assert True

    def test_stow_store_instances_success(self, tmp_path):
        dcm_file = tmp_path / "test.dcm"
        dcm_file.write_bytes(b"DICOM_STREAM_DATA")
        client = DICOMwebClient()

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "ID": "orthanc_12345",
                "ParentStudy": "study_67890",
            }
            res = client.stow_store_instances(str(dcm_file))
            assert res["status"] == "success"
            assert res["orthanc_id"] == "orthanc_12345"
            assert res["parent_study"] == "study_67890"

    def test_get_ohif_viewer_url(self):
        url = dicomweb_client.get_ohif_viewer_url("1.2.3.4.5")
        assert "ohif-viewer/viewer" in url
        assert "1.2.3.4.5" in url


class TestDICOMServicePACSIntegration:
    """Test DICOMService integration with STOW-RS push."""

    def test_store_dicom_with_stow_push(self, app, tmp_path):
        from departments.imaging.dicom_service import DICOMService

        with app.app_context():
            _create_test_patient("PTPACS01", "PACS Test Patient")

            dcm_path = tmp_path / "sample.dcm"
            dcm_path.write_bytes(b"HEADER_DICOM")

            mock_ds = MagicMock()
            mock_ds.PatientID = "PTPACS01"
            mock_ds.PatientName = "PACS Test Patient"
            mock_ds.StudyInstanceUID = "1.2.840.10008.5.1.4.1.1.1.100"
            mock_ds.SeriesInstanceUID = "1.2.840.10008.5.1.4.1.1.1.200"
            mock_ds.SOPInstanceUID = "1.2.840.10008.5.1.4.1.1.1.300"
            mock_ds.Modality = "CT"
            mock_ds.StudyDate = "20260913"
            mock_ds.StudyDescription = "Chest CT Scan"
            mock_ds.SeriesDescription = "Axial 5mm"
            mock_ds.AccessionNumber = "ACC999"
            mock_ds.BodyPartExamined = "CHEST"

            with patch("pydicom.dcmread", return_value=mock_ds):
                with patch.object(dicomweb_client, "stow_store_instances") as mock_stow:
                    mock_stow.return_value = {
                        "status": "success",
                        "orthanc_id": "orthanc_chest_ct_100",
                        "parent_study": "study_100",
                    }
                    res = DICOMService.store_dicom(str(dcm_path))

                    assert res.result_id == "1.2.840.10008.5.1.4.1.1.1.300"
                    assert res.patient_id == "PTPACS01"
                    assert res.orthanc_uid == "orthanc_chest_ct_100"
                    assert res.storage_backend == "orthanc_pacs"


class TestOHIFAndDICOMwebRoutes:
    """Test OHIF Viewer & DICOMweb Proxy Routes."""

    def test_ohif_viewer_route(self, client, app, admin_user):
        with app.app_context():
            _create_test_patient("PTOHIF01", "OHIF Patient", sex="Female")

            img = Imaging(imaging_type="Brain MRI", cost=4500.0)
            db.session.add(img)
            db.session.commit()

            res = ImagingResult(
                result_id="sop.ohif.mri.001",
                patient_id="PTOHIF01",
                imaging_id=img.id,
                orthanc_uid="orthanc_mri_001",
                processing_metadata={
                    "study_instance_uid": "study.ohif.mri.001",
                    "modality": "MR",
                },
            )
            db.session.add(res)
            db.session.commit()

        resp = client.get("/imaging/dicom/ohif/sop.ohif.mri.001")
        assert resp.status_code == 200
        assert b"OHIF DICOM Workstation" in resp.data
        assert b"PTOHIF01" in resp.data
        assert b"orthanc_mri_001" in resp.data

    def test_qido_search_proxy(self, client, admin_user):
        with patch.object(dicomweb_client, "qido_search_studies", return_value=[{"study": "1"}]):
            resp = client.get("/imaging/dicom/qido?PatientID=PT001")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["count"] == 1

    def test_wado_metadata_proxy(self, client, admin_user):
        with patch.object(dicomweb_client, "wado_retrieve_metadata", return_value=[{"meta": "1"}]):
            resp = client.get("/imaging/dicom/wado/1.2.3.4")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["study_instance_uid"] == "1.2.3.4"
            assert len(data["metadata"]) == 1


class TestFHIRImagingStudyEndpoint:
    """Test FHIR R4 ImagingStudy resource endpoints."""

    def test_fhir_imaging_study_get(self, client, app, admin_user):
        with app.app_context():
            _create_test_patient("PTFHIRIMG01", "FHIR Img Patient")

            img = Imaging(imaging_type="Chest X-Ray", cost=2000.0)
            db.session.add(img)
            db.session.commit()

            res = ImagingResult(
                result_id="sop.fhir.xray.001",
                patient_id="PTFHIRIMG01",
                imaging_id=img.id,
                processing_metadata={
                    "study_instance_uid": "study.fhir.xray.001",
                    "modality": "DX",
                },
            )
            db.session.add(res)
            db.session.commit()

        resp = client.get("/api/fhir/R4/ImagingStudy/sop.fhir.xray.001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["resourceType"] == "ImagingStudy"
        assert data["subject"]["reference"] == "Patient/PTFHIRIMG01"
        assert data["modality"][0]["code"] == "DX"

    def test_fhir_imaging_study_search(self, client, app, admin_user):
        with app.app_context():
            _create_test_patient("PTFHIRIMG01", "FHIR Img Patient")

            img = Imaging(imaging_type="Ultrasound Abdomen", cost=1800.0)
            db.session.add(img)
            db.session.commit()

            res = ImagingResult(
                result_id="sop.fhir.search.001",
                patient_id="PTFHIRIMG01",
                imaging_id=img.id,
                processing_metadata={
                    "study_instance_uid": "study.fhir.search.001",
                    "modality": "US",
                },
            )
            db.session.add(res)
            db.session.commit()

        resp = client.get("/api/fhir/R4/ImagingStudy?patient=PTFHIRIMG01")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["resourceType"] == "Bundle"
        assert data["type"] == "searchset"
        assert data["total"] >= 1
