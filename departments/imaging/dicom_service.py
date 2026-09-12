"""
departments/imaging/dicom_service.py
─────────────────────────────────────
Phase 5 — DICOM PACS Integration
"""

import logging
import os
from pathlib import Path
from typing import Optional

from departments.models.imaging import ImagingResult
from departments.models.records import Patient
from extensions import db

logger = logging.getLogger(__name__)

# DICOM storage configuration
DICOM_STORAGE_BASE = Path(os.getenv("DICOM_STORAGE_PATH", "storage/dicom"))
DICOM_INCOMING = DICOM_STORAGE_BASE / "incoming"
DICOM_ARCHIVE = DICOM_STORAGE_BASE / "archive"
DICOM_THUMBNAILS = DICOM_STORAGE_BASE / "thumbnails"

# Ensure directories exist
DICOM_INCOMING.mkdir(parents=True, exist_ok=True)
DICOM_ARCHIVE.mkdir(parents=True, exist_ok=True)
DICOM_THUMBNAILS.mkdir(parents=True, exist_ok=True)


class DICOMService:
    """Handles DICOM file operations and PACS integration."""

    @staticmethod
    def parse_dicom(file_path: str) -> dict:
        """Parse a DICOM file and extract relevant metadata."""
        try:
            from pydicom import dcmread

            ds = dcmread(file_path)

            metadata = {
                "patient_id": str(getattr(ds, "PatientID", "")),
                "patient_name": str(getattr(ds, "PatientName", "")),
                "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
                "modality": str(getattr(ds, "Modality", "")),
                "study_date": str(getattr(ds, "StudyDate", "")),
                "study_description": str(getattr(ds, "StudyDescription", "")),
                "series_description": str(getattr(ds, "SeriesDescription", "")),
                "accession_number": str(getattr(ds, "AccessionNumber", "")),
                "body_part": str(getattr(ds, "BodyPartExamined", "")),
            }

            return metadata

        except Exception as e:
            logger.exception("Failed to parse DICOM file: %s", file_path)
            raise ValueError(f"Invalid DICOM file: {e}")

    @staticmethod
    def store_dicom(file_path: str, imaging_id: Optional[int] = None) -> ImagingResult:
        """Store a DICOM file and create an ImagingResult record."""
        metadata = DICOMService.parse_dicom(file_path)

        sop_uid = metadata["sop_instance_uid"]
        if not sop_uid:
            raise ValueError("DICOM file missing SOPInstanceUID")

        study_uid = metadata["study_instance_uid"]
        storage_dir = DICOM_ARCHIVE / study_uid
        storage_dir.mkdir(parents=True, exist_ok=True)

        dest_path = storage_dir / f"{sop_uid}.dcm"
        os.rename(file_path, dest_path)

        # Check if result already exists
        imaging_result = ImagingResult.query.filter_by(result_id=sop_uid).first()

        if not imaging_result:
            # Find patient if patient_id is available
            patient = None
            patient_id = metadata["patient_id"]
            if patient_id:
                patient = Patient.query.filter_by(patient_id=patient_id).first()

            if not patient:
                raise ValueError(f"Patient {patient_id} not found")

            # Find or create Imaging record
            from departments.models.medicine import Imaging
            imaging = None
            if imaging_id:
                imaging = db.session.get(Imaging, imaging_id)

            if not imaging:
                # Create a generic imaging record
                imaging = Imaging(
                    patient_id=patient.patient_id,
                    test_name=metadata["study_description"] or f"{metadata['modality']} Study",
                    status="completed",
                )
                db.session.add(imaging)
                db.session.flush()

            imaging_result = ImagingResult(
                result_id=sop_uid,
                patient_id=patient.patient_id,
                imaging_id=imaging.id,
                dicom_file_path=str(dest_path),
                result_notes=f"DICOM: {metadata['study_description']} - {metadata['series_description']}",
                processing_metadata={
                    "sop_instance_uid": sop_uid,
                    "study_instance_uid": study_uid,
                    "series_instance_uid": metadata["series_instance_uid"],
                    "modality": metadata["modality"],
                    "study_date": metadata["study_date"],
                    "accession_number": metadata["accession_number"],
                    "body_part": metadata["body_part"],
                },
                files_processed=1,
            )

            db.session.add(imaging_result)
            db.session.commit()

            logger.info("Stored DICOM file: %s -> %s", file_path, dest_path)

        return imaging_result

    @staticmethod
    def get_dicom_by_sop_uid(sop_instance_uid: str) -> Optional[str]:
        """Retrieve the file path for a DICOM file by SOP Instance UID."""
        result = ImagingResult.query.filter_by(result_id=sop_instance_uid).first()

        if result and result.dicom_file_path:
            if Path(result.dicom_file_path).exists():
                return result.dicom_file_path

        return None

    @staticmethod
    def list_studies(patient_id: str) -> list[dict]:
        """List all studies for a patient."""
        results = ImagingResult.query.filter_by(patient_id=patient_id).all()

        studies = {}
        for result in results:
            metadata = result.processing_metadata or {}
            study_uid = metadata.get("study_instance_uid", result.result_id)

            if study_uid not in studies:
                studies[study_uid] = {
                    "study_instance_uid": study_uid,
                    "study_date": metadata.get("study_date", ""),
                    "study_description": metadata.get("study_description", ""),
                    "modality": metadata.get("modality", ""),
                    "series_count": 0,
                    "instances": [],
                }

            studies[study_uid]["series_count"] += 1
            studies[study_uid]["instances"].append({
                "sop_instance_uid": result.result_id,
                "series_instance_uid": metadata.get("series_instance_uid", ""),
                "modality": metadata.get("modality", ""),
            })

        return list(studies.values())
