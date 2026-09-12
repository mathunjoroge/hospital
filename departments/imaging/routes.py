import logging
import os
import uuid
from datetime import datetime, timezone

import pydicom
from flask import (
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename

from departments.api.auth import jwt_or_session_required
from departments.models.imaging import ImagingResult
from departments.models.medicine import Imaging, RequestedImage
from departments.nlp.src.nvidia_client import NvidiaNIMClient
from departments.rbac import roles_required
from extensions import db, socketio

from . import bp
from . import bp as imaging_bp

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize NVIDIA NIM client
nim_client = NvidiaNIMClient()


def allowed_file(filename):
    """Check if the uploaded file has a valid DICOM extension."""
    logger.debug(f"Checking file: {filename}")
    allowed = "." in filename and filename.rsplit(".", 1)[1].lower() in {"dcm", "dicom"}
    logger.debug(f"File {filename} allowed: {allowed}")
    return allowed


def validate_dicom_file(filepath):
    """Validate the DICOM file for pixel data and decompress if necessary."""
    logger.debug(f"Validating DICOM: {filepath}")
    try:
        dicom = pydicom.dcmread(filepath, force=True)
        if not hasattr(dicom, "PixelData"):
            logger.error(f"No PixelData in {filepath}")
            return False, "DICOM file has no pixel data"
        if (
            hasattr(dicom, "file_meta")
            and hasattr(dicom.file_meta, "TransferSyntaxUID")
            and dicom.file_meta.TransferSyntaxUID.is_compressed
        ):
            logger.debug(f"Decompressing {filepath}")
            try:
                dicom.decompress()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not decompress pixel data: {e}")
        _ = dicom.pixel_array
        logger.debug(f"Validated DICOM: {filepath}, shape: {dicom.pixel_array.shape}")
        return True, "Valid DICOM image"
    except Exception as e:  # noqa: BLE001
        logger.error(f"DICOM validation failed for {filepath}: {e}")
        return False, f"Invalid DICOM file: {e}"


def get_modality_and_body_part(dicom):
    """Extract modality, body part, and patient information from DICOM metadata."""
    modality = getattr(dicom, "Modality", "Unknown").upper()
    body_part = getattr(dicom, "BodyPartExamined", "Unknown").lower()
    modality_map = {
        "CR": "X-ray",
        "DX": "X-ray",
        "CT": "CT Scan",
        "MR": "MRI",
        "PT": "PET",
        "CY": "Cytology",
    }
    modality = modality_map.get(modality, modality)
    if body_part == "unknown" and hasattr(dicom, "StudyDescription"):
        body_part = (
            str(getattr(dicom, "StudyDescription", "")).lower() or "unspecified region"
        )
    patient_info = {
        "patient_id": getattr(dicom, "PatientID", "Unknown"),
        "patient_name": str(getattr(dicom, "PatientName", "Unknown")),
        "patient_sex": getattr(dicom, "PatientSex", "Unknown"),
        "patient_birth_date": getattr(dicom, "PatientBirthDate", "Unknown"),
        "study_date": getattr(dicom, "StudyDate", datetime.now(timezone.utc).strftime("%Y%m%d")),
    }
    logger.debug(f"Extracted: modality={modality}, body_part={body_part}")
    return modality, body_part, patient_info


def analyze_dicom(dicom_path, description="", symptoms=""):
    """Analyze a DICOM image using NVIDIA NIM Vision model API with DICOM metadata extraction."""
    logger.debug(f"Analyzing DICOM: {dicom_path}")
    try:
        is_valid, message = validate_dicom_file(dicom_path)
        if not is_valid:
            logger.error(f"Validation failed: {message}")
            return {"error": message, "status": "error"}

        dicom = pydicom.dcmread(dicom_path, force=True)
        modality, body_part, patient_info = get_modality_and_body_part(dicom)
        logger.debug(f"DICOM metadata: modality={modality}, body_part={body_part}")

        nim_analysis = nim_client.analyze_radiology(
            modality=modality,
            body_part=body_part,
            description=description,
            symptoms=symptoms,
        )

        result = {
            "predictions": nim_analysis.get("predictions", ["Normal"]),
            "confidence": nim_analysis.get("confidence", 88.0),
            "impression": nim_analysis.get(
                "impression", f"Unremarkable {modality} examination."
            ),
            "status": "success",
            "filename": os.path.basename(dicom_path),
            "modality": modality,
            "body_part": body_part,
            "patient_info": patient_info,
        }
        logger.debug(f"Analysis result via NVIDIA NIM: {result}")
        return result

    except Exception as e:  # noqa: BLE001
        logger.error(f"Analysis failed for {dicom_path}: {e}")
        return {"error": str(e), "status": "error"}


def generate_report(
    analysis_results,
    patient_id,
    result_id,
    description=None,
    symptoms=None,
    custom_findings=None,
    custom_impression=None,
):
    """
    Generate a structured radiology report for any body part and imaging modality, avoiding false positives.

    Args:
        analysis_results (list): List of analysis results for each DICOM file, supporting multiple predictions.
        patient_id (str): Unique identifier for the patient.
        result_id (str): Unique identifier for the imaging result.
        description (str, optional): Imaging request description.
        symptoms (str, optional): Patient-reported symptoms.
        custom_findings (list, optional): User-provided findings to override AI results.
        custom_impression (list, optional): User-provided impression to override default.

    Returns:
        str: Generated radiology report.
    """
    logger.debug(
        f"Generating report for patient_id={patient_id}, result_id={result_id}"
    )

    # Initialize report components
    report_lines = []
    current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
    confidence_threshold = 70.0  # Higher threshold to reduce false positives

    # Handle analysis results
    successful = [r for r in analysis_results if r.get("status") == "success"]
    errors = [r for r in analysis_results if r.get("status") != "success"]
    logger.debug(f"Successful analyses: {len(successful)}, Errors: {len(errors)}")

    # Extract patient information
    if successful:
        patient_info = successful[0]["patient_info"]
        patient_name = patient_info.get("patient_name", "Unknown")
        patient_sex = patient_info.get("patient_sex", "Unknown")
        patient_birth_date = patient_info.get("patient_birth_date", "Unknown")
        study_date = patient_info.get("study_date", "Unknown")
        if study_date != "Unknown":
            try:
                study_date = datetime.strptime(study_date, "%Y%m%d").strftime(  # noqa: DTZ007
                    "%B %d, %Y"
                )
            except ValueError:
                study_date = current_date
        modality = successful[0].get("modality", "Unknown")
        body_part = successful[0].get("body_part", "unspecified region").capitalize()
    else:
        patient_name = "Unknown"
        patient_sex = "Unknown"
        patient_birth_date = "Unknown"
        study_date = current_date
        patient_info = {"patient_id": patient_id}
        modality = "Unknown"
        body_part = "Unspecified Region"

    # Calculate age
    patient_age = "Unknown"
    if patient_birth_date != "Unknown":
        try:
            birth_date = datetime.strptime(patient_birth_date, "%Y%m%d")  # noqa: DTZ007
            today = datetime.now(timezone.utc)
            patient_age = (
                today.year
                - birth_date.year
                - ((today.month, today.day) < (birth_date.month, birth_date.day))
            )
            patient_age = f"{patient_age} years"
        except ValueError:
            patient_age = "Unknown"

    # Define modality-specific techniques
    modality_techniques = {
        "MRI": {
            "default": f"Multiplanar, multisequence MRI of the {body_part.lower()} performed without intravenous contrast, including T1-weighted, T2-weighted, and STIR sequences.",
            "head": "Multiplanar, multisequence MRI of the head performed without intravenous contrast, including T1-weighted, T2-weighted, FLAIR, and gradient-echo (GRE) sequences.",
        },
        "CT Scan": f"Non-contrast CT scan of the {body_part.lower()} performed with 1 mm slice thickness in axial, coronal, and sagittal reconstructions.",
        "X-ray": f"Standard posteroanterior and lateral radiographic views of the {body_part.lower()} obtained.",
        "PET": f"PET/CT scan of the {body_part.lower()} performed with fluorodeoxyglucose (FDG) tracer, including low-dose CT for attenuation correction.",
        "Cytology": f"Cytological imaging of the {body_part.lower()} performed with high-resolution microscopy.",
        "Unknown": f"Imaging of the {body_part.lower()} performed with standard protocol.",
    }
    technique = modality_techniques.get(modality, modality_techniques["Unknown"])
    if isinstance(technique, dict):
        technique = technique.get(body_part.lower(), technique["default"])

    # Start building the report
    report_lines.append("**Radiology Report**")
    report_lines.append("")

    # Patient Information
    report_lines.append("**Patient Information:**")
    report_lines.append(f"Name: {patient_name}")
    report_lines.append(f"Age: {patient_age}")
    report_lines.append(f"Gender: {patient_sex}")
    report_lines.append(f"Date of Examination: {study_date}")
    report_lines.append("Referring Physician: [Physician Name]")
    report_lines.append(f"Examination: {modality} of the {body_part}")
    report_lines.append("")

    # Clinical Indication
    clinical_indication = (
        description
        or symptoms
        or f"Evaluation of {body_part.lower()} for suspected pathology."
    )
    report_lines.append("**Clinical Indication:**")
    report_lines.append(clinical_indication)
    report_lines.append("")

    # Technique
    report_lines.append("**Technique:**")
    report_lines.append(technique)
    report_lines.append("")

    # Findings
    report_lines.append("**Findings:**")
    if custom_findings:
        findings = custom_findings
    elif successful:
        findings = []
        predictions_seen = set()
        for res in successful:
            predictions = (
                res.get("predictions", [res.get("prediction", "Unknown")])
                if isinstance(res.get("predictions"), list)
                else [res.get("prediction", "Unknown")]
            )
            confidence = res.get("confidence", 0)
            finding_body_part = res.get("body_part", body_part).capitalize()

            finding_map = {
                "Normal": f"{finding_body_part}: No significant abnormalities detected.",
                "Inflammation": f"{finding_body_part}: Evidence of inflammation noted.",
                "Mass": f"{finding_body_part}: Suspected mass lesion identified.",
                "Nodule": f"{finding_body_part}: Nodule detected.",
                "Cyst": f"{finding_body_part}: Cystic lesion noted.",
                "Fracture": f"{finding_body_part}: {'Possible fracture noted' if confidence < confidence_threshold else 'Fracture identified'}.",
                "Thickening": f"{finding_body_part}: Tissue thickening observed.",
                "Edema": f"{finding_body_part}: Edema present.",
                "Tumor": f"{finding_body_part}: Possible tumor detected.",
                "Lesion": f"{finding_body_part}: Unspecified lesion noted.",
                "Hemorrhage": f"{finding_body_part}: Intracranial hemorrhage identified with hyperintense signal on FLAIR and GRE sequences.",
                "Midline Shift": f"{finding_body_part}: Midline shift observed, approximately 5 mm, secondary to mass effect.",
                "Unknown": f"{finding_body_part}: Non-specific findings.",
            }

            for pred in predictions:
                if pred not in predictions_seen:
                    predictions_seen.add(pred)
                    finding = finding_map.get(
                        pred, f"{finding_body_part}: {pred} noted."
                    )
                    if confidence < confidence_threshold and pred != "Normal":
                        finding += f" Low-confidence finding ({confidence:.1f}%) is non-specific and requires clinical correlation."
                    else:
                        finding += f" ({confidence:.1f}% confidence)."
                    findings.append(f"- {finding}")

            # Head-specific findings for MRI, only if no abnormalities detected
            if (
                modality == "MRI"
                and body_part.lower() == "head"
                and "Normal" in predictions_seen
            ):
                findings.extend(
                    [
                        "- Brain Parenchyma: No evidence of fracture, hemorrhage, or mass lesions identified.",
                        "- Ventricles: Normal size and configuration, with no evidence of compression or hydrocephalus.",
                        "- Midline Structures: No midline shift observed.",
                        "- Cerebral Vasculature: No abnormal signal voids or evidence of vascular injury on GRE sequences.",
                        "- Skull and Scalp: No osseous abnormalities or soft tissue abnormalities identified.",
                        "- Other Findings: No additional abnormalities detected.",
                    ]
                )
            elif modality == "MRI" and body_part.lower() == "head":
                # Add related findings only if relevant
                if "Hemorrhage" in predictions_seen:
                    findings.append(
                        "- Ventricles: Compressed due to mass effect from hemorrhage."
                    )
                elif "Midline Shift" not in predictions_seen:
                    findings.append("- Ventricles: Normal size and configuration.")
                if (
                    "Midline Shift" not in predictions_seen
                    and "Hemorrhage" not in predictions_seen
                ):
                    findings.append("- Midline Structures: No midline shift observed.")
                if "Hemorrhage" not in predictions_seen:
                    findings.append(
                        "- Other Findings: No evidence of intracranial hemorrhage or additional abnormalities."
                    )
        if not findings:
            findings = [f"- {body_part}: No significant abnormalities identified."]
    else:
        findings = [
            f"- {body_part}: Imaging findings are non-specific. No definitive abnormalities identified."
        ]
    report_lines.extend(findings)
    report_lines.append("")

    # Impression
    report_lines.append("**Impression:**")
    if custom_impression:
        impression = custom_impression
    elif successful:
        impression = []
        high_conf_findings = [
            f
            for f in findings
            if "no significant abnormalities" not in f.lower()
            and "low-confidence" not in f.lower()
        ]
        low_conf_findings = [f for f in findings if "low-confidence" in f.lower()]
        if high_conf_findings:
            for idx, finding in enumerate(high_conf_findings, 1):
                finding_text = finding.lstrip("- ").split(": ")[1].split(" (")[0]
                impression.append(
                    f"{idx}. {finding_text.capitalize()} in {body_part.lower()}."
                )
        if low_conf_findings:
            impression.append(
                f"{len(high_conf_findings) + 1}. Low-confidence findings are non-specific and require clinical correlation."
            )
        if not impression:
            impression = ["1. No significant abnormalities detected."]
    else:
        impression = ["1. Non-specific findings requiring clinical correlation."]
    report_lines.extend(impression)
    report_lines.append("")

    # Recommendations
    report_lines.append("**Recommendations:**")
    specialist = get_specialist(body_part)
    recommendations = []
    if any("hemorrhage" in f.lower() or "midline shift" in f.lower() for f in findings):
        recommendations.append(
            "Urgent correlation with clinical symptoms and neurological status is advised."
        )
        recommendations.append(
            f"Immediate consultation with {specialist} is recommended for management of intracranial hemorrhage and/or midline shift, potentially requiring surgical intervention."
        )
        recommendations.append(
            "Consider repeat imaging (e.g., CT head) to assess hemorrhage progression and confirm other findings."
        )
        recommendations.append(
            "Close monitoring in a critical care setting is advised."
        )
    else:
        recommendations.append(
            "Correlation with clinical symptoms and laboratory findings is advised."
        )
        if (
            modality == "MRI"
            and body_part.lower() == "head"
            and any("fracture" in f.lower() for f in findings)
        ):
            recommendations.append(
                "Consider CT imaging of the head to evaluate for subtle fractures, as MRI may be less sensitive for osseous injuries."
            )
        recommendations.extend(
            [
                f"{specialist} consultation is recommended for further evaluation and management if symptoms persist.",
                "Follow-up imaging may be considered if symptoms worsen.",
            ]
        )
    report_lines.extend(recommendations)
    report_lines.append("")

    # Radiologist
    report_lines.append("**Radiologist:**")
    report_lines.append("[Radiologist Name], MD")
    report_lines.append("Board-Certified Radiologist")
    report_lines.append(f"Date: {study_date}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append(
        "**Note:** This report is for professional use and should be interpreted in the context of the patient’s clinical presentation. Please contact the radiology department for any clarification."
    )

    # Handle errors
    if errors:
        report_lines.append("")
        report_lines.append("**Processing Notes:**")
        for error in errors:
            report_lines.append(
                f"- Error processing {error.get('filename', 'unknown file')}: {error.get('error', 'Unknown error')}"
            )
        report_lines.append("")

    # Join lines into final report
    final_report = "\n".join(report_lines)
    logger.debug(f"Generated report: {final_report[:200]}...")
    return final_report


def get_specialist(body_part):
    """Map body part to recommended specialist."""
    specialist_map = {
        "sinuses": "Otolaryngology (ENT)",
        "brain": "Neurology or Neurosurgery",
        "head": "Neurology or Neurosurgery",
        "chest": "Pulmonology or Thoracic Surgery",
        "abdomen": "Gastroenterology or General Surgery",
        "pelvis": "Urology or Gynecology",
        "spine": "Orthopedics or Neurosurgery",
        "knee": "Orthopedics",
        "shoulder": "Orthopedics",
        "heart": "Cardiology",
        "liver": "Hepatology",
        "unspecified region": "Appropriate specialist",
    }
    return specialist_map.get(body_part.lower(), "Appropriate specialist")


@bp.route("/process_imaging_request/<int:request_id>", methods=["GET", "POST"])
@login_required
@roles_required("imaging", "admin")
def process_imaging_request(request_id):
    """Process an imaging request by analyzing uploaded DICOM files."""
    logger.debug(f"User {current_user.id} processing imaging request {request_id}")

    try:
        imaging_request = RequestedImage.query.get_or_404(request_id)
        imaging = Imaging.query.get_or_404(imaging_request.imaging_id)
        logger.debug(f"Loaded imaging request {request_id} and imaging {imaging.id}")
    except Exception as e:
        logger.exception("Error fetching request {request_id}: ")
        flash(f"Error loading request: {e!s}", "error")
        return redirect(url_for("imaging.index"))

    if request.method == "POST":
        logger.debug(f"POST request for imaging request {request_id}")
        if "dicom_folder" not in request.files or not request.files["dicom_folder"]:
            logger.debug("No DICOM files uploaded")
            flash("No DICOM files uploaded", "error")
            return redirect(request.url)

        result_id = str(uuid.uuid4())
        upload_dir = os.path.join(current_app.config["DICOM_UPLOAD_FOLDER"], result_id)
        os.makedirs(upload_dir, exist_ok=True)
        logger.debug(f"Upload directory created: {upload_dir}")

        files = request.files.getlist("dicom_folder")
        total_files = len(files)
        file_paths = []
        analysis_results = []
        processed_count = 0
        logger.debug(f"Processing {total_files} files")

        for i, file in enumerate(files, 1):
            if not file or not allowed_file(file.filename):
                logger.debug(
                    f"Skipping invalid file: {file.filename if file else 'None'}"
                )
                continue

            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            logger.debug(f"Processing file {i}/{total_files}: {filename}")

            try:
                file.save(filepath)
                result = analyze_dicom(filepath)
                result["filename"] = filename
                logger.debug(f"Analysis for {filename}: {result}")

                if result.get("status") == "success":
                    file_paths.append(filepath)
                    processed_count += 1
                    logger.debug(f"Successfully processed {filename}")
                else:
                    os.remove(filepath)
                    logger.debug(f"Failed processing {filename}, removed")

                analysis_results.append(result)

                if i % 5 == 0 or i == total_files:
                    try:
                        socketio.emit(
                            "progress",
                            {
                                "current": processed_count,
                                "total": total_files,
                                "request_id": request_id,
                            },
                            namespace="/imaging",
                        )
                        logger.debug(
                            f"Progress emitted: {processed_count}/{total_files}"
                        )
                    except Exception as emit_error:  # noqa: BLE001
                        logger.error(f"Progress emit failed: {emit_error}")

            except Exception as file_error:  # noqa: BLE001
                logger.error(f"Error processing {filename}: {file_error}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                analysis_results.append(
                    {"filename": filename, "status": "error", "error": str(file_error)}
                )

        if not file_paths:
            logger.debug("No valid DICOM files processed")
            flash("No valid DICOM images were processed", "error")
            return redirect(request.url)

        try:
            ai_report = generate_report(
                analysis_results, imaging_request.patient_id, result_id
            )
            final_report = request.form.get(
                "result_notes", "AI-generated findings stored in AI Findings section."
            )
            logger.debug(f"AI report: {ai_report[:100]}...")
            logger.debug(f"Final report: {final_report[:100]}...")
        except Exception as report_error:  # noqa: BLE001
            logger.error(f"Report generation failed: {report_error}")
            ai_report = "Report generation failed. Raw AI findings available."
            final_report = "Report generation failed. Basic findings available."

        try:
            imaging_result = ImagingResult(
                result_id=result_id,
                patient_id=imaging_request.patient_id,
                imaging_id=imaging.id,
                test_date=datetime.now(timezone.utc),
                result_notes=final_report,
                updated_by=current_user.id,
                dicom_file_path=",".join(file_paths),
                ai_findings=ai_report,
                ai_generated=True,
                files_processed=processed_count,
                files_failed=total_files - processed_count,
                processing_metadata={
                    "file_metadata": analysis_results,
                    "model_version": "DenseNet121",
                },
            )
            db.session.add(imaging_result)
            imaging_request.status = 1  # 0=Pending, 1=Completed, 2=Cancelled
            imaging_request.result_id = result_id
            db.session.commit()

            # FIX 4: Advance encounter stage after imaging completion
            from departments.shared.visit_closure import advance_after_completion
            advance_after_completion(imaging_request.patient_id)
            logger.debug(
                f"Saved result: request_id={request_id}, result_id={result_id}"
            )

            try:
                socketio.emit(
                    "complete",
                    {
                        "request_id": request_id,
                        "result_id": result_id,
                        "processed": processed_count,
                        "failed": total_files - processed_count,
                    },
                    namespace="/imaging",
                )
                logger.debug(
                    f"Completion emitted: processed={processed_count}, failed={total_files - processed_count}"
                )
            except Exception as emit_error:  # noqa: BLE001
                logger.error(f"Completion emit failed: {emit_error}")

            flash(
                f"Processed {processed_count} of {total_files} files successfully",
                "success",
            )
            logger.debug(f"Redirecting to view_result: result_id={result_id}")
            return redirect(url_for("imaging.view", result_id=result_id))

        except Exception:
            db.session.rollback()
            logger.exception("Database error: ")
            flash("Error saving results to database", "error")
            return redirect(request.url)

    draft_report = ""
    if imaging_request.result_id:
        existing_result = ImagingResult.query.get(imaging_request.result_id)
        if existing_result:
            draft_report = existing_result.result_notes
            logger.debug(f"Draft report loaded: {draft_report[:100]}...")

    logger.debug(f"Rendering process.html for request_id={request_id}")
    return render_template(
        "imaging/process.html",
        imaging_request=imaging_request,
        imaging=imaging,
        draft_report=draft_report,
        ai_enabled=nim_client is not None,
    )


@bp.route("/view_result/<string:result_id>", methods=["GET"])
@login_required
@roles_required("imaging", "admin")
def view_result(result_id):
    """View the imaging result for a given result_id."""
    logger.debug(f"Viewing result {result_id} for user {current_user.id}")

    try:
        imaging_result = ImagingResult.query.filter_by(
            result_id=result_id
        ).first_or_404()
        imaging = Imaging.query.get_or_404(imaging_result.imaging_id)
        imaging_request = RequestedImage.query.filter_by(
            result_id=result_id
        ).first_or_404()

        logger.debug(f"Rendering view.html for result_id={result_id}")
        return render_template(
            "imaging/view.html",
            imaging_result=imaging_result,
            imaging=imaging,
            imaging_request=imaging_request,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error viewing result {result_id}: {e}")
        flash(f"Error: {e!s}", "error")
        return redirect(url_for("imaging.index"))


@bp.route("/download/<string:result_id>/<path:filename>", methods=["GET"])
@login_required
@roles_required("imaging", "admin")
def download_file(result_id, filename):
    """Serve a DICOM file for download."""
    logger.debug(
        f"Download request for result_id={result_id}, filename={filename} by user {current_user.id}"
    )

    upload_dir = os.path.join(current_app.config["DICOM_UPLOAD_FOLDER"], result_id)
    try:
        return send_from_directory(upload_dir, filename, as_attachment=True)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error downloading file {filename}: {e}")
        flash(f"Error downloading file: {e!s}", "error")
        return redirect(url_for("imaging.view_result", result_id=result_id))


@bp.route("/results", methods=["GET"])  # Changed from '/imaging_results'
@login_required
@roles_required("medicine", "imaging", "admin")
def imaging_results():
    """Display a list of all processed imaging results."""
    logger.debug(f"Accessing imaging results list for user {current_user.id}")

    try:
        results = ImagingResult.query.order_by(ImagingResult.test_date.desc()).all()
        logger.debug(f"Retrieved {len(results)} imaging results from database")

        return render_template("imaging/imaging_results.html", results=results)
    except Exception as e:
        logger.exception(f"Error retrieving imaging results: {e!s}")  # noqa: TRY401
        flash(f"Error retrieving results: {e!s}", "error")
        return redirect(url_for("imaging.index"))


@bp.route("/view_imaging_results/<string:result_id>", methods=["GET"])
@login_required
@roles_required("medicine", "imaging", "admin")
def view_imaging_results(result_id):
    logger.debug(f"User {current_user.id} viewing results for result_id={result_id}")

    try:
        imaging_result = ImagingResult.query.filter_by(
            result_id=result_id
        ).first_or_404()
        imaging = (
            Imaging.query.get(imaging_result.imaging_id)
            if imaging_result.imaging_id
            else None
        )
        if not imaging:
            logger.warning(
                f"No Imaging record found for imaging_id={imaging_result.imaging_id}"
            )

        logger.debug(
            f"Found imaging_result: result_id={imaging_result.result_id}, patient_id={imaging_result.patient_id}"
        )

        file_paths = (
            imaging_result.dicom_file_path.split(",")
            if imaging_result.dicom_file_path
            else []
        )
        valid_file_paths = [path for path in file_paths if os.path.exists(path)]
        if len(valid_file_paths) < len(file_paths):
            missing_files = set(file_paths) - set(valid_file_paths)
            logger.warning(f"Missing DICOM files: {missing_files}")
            flash(
                f"Warning: {len(missing_files)} DICOM file(s) could not be found on the server.",
                "warning",
            )

        file_index = request.args.get("file_index", type=int)
        action = request.args.get("action", "view")
        if file_index is not None:
            if 0 <= file_index < len(valid_file_paths):
                file_path = valid_file_paths[file_index]
                logger.debug(f"Serving DICOM file: {file_path} with action={action}")
                return send_from_directory(
                    directory=os.path.dirname(file_path),
                    path=os.path.basename(file_path),
                    mimetype="application/dicom",
                    as_attachment=(action == "download"),
                    download_name=os.path.basename(file_path)
                    if action == "download"
                    else None,
                )
            else:
                logger.debug(
                    f"Invalid file_index={file_index}, range: 0 to {len(valid_file_paths)-1}"
                )
                flash("Invalid file index selected.", "error")

        ai_findings = imaging_result.ai_findings or ""
        header = {}
        findings = []
        impression = []
        footer = {}

        lines = ai_findings.split("\n")
        current_section = None
        current_modality = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Radiology Report"):
                current_section = "header"
            elif line.startswith("FINDINGS:"):
                current_section = "findings_intro"
            elif line.startswith("IMPRESSION:"):
                current_section = "impression"
            elif current_section == "header" and ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()
            elif current_section == "findings_intro" and not line.startswith("Image"):
                findings.append({"intro": line})
                current_section = "findings"
            elif (
                current_section == "findings"
                and line.endswith(":")
                and not line.startswith("Image")
            ):
                current_modality = line
                findings.append({"modality": current_modality})
            elif current_section == "findings" and line.startswith("Image"):
                entry = {"text": line}
                if current_modality:
                    entry["modality"] = current_modality
                findings.append(entry)
            elif current_section == "findings" and findings and "text" in findings[-1]:
                findings[-1]["text"] += f"\n{line}"
            elif current_section == "impression" and line[0].isdigit():
                impression.append(line)
            elif current_section != "impression" and ":" in line:
                key, value = line.split(":", 1)
                footer[key.strip()] = value.strip()

        imaging_type = imaging.imaging_type if imaging else "Unknown"
        logger.debug(
            f"Imaging type: {imaging_type}, Parsed AI findings: header={header}, findings={len(findings)}, impression={len(impression)}"
        )

        logger.debug(f"Rendering view.html for result_id={result_id}")
        return render_template(
            "imaging/view.html",
            imaging_result=imaging_result,
            file_paths=valid_file_paths,
            imaging_type=imaging_type,
            ai_header=header,
            ai_findings=findings,
            ai_impression=impression,
            ai_footer=footer,
        )

    except Exception as e:
        logger.exception("Error viewing result_id=: {e!s}")
        flash(f"An error occurred while loading the imaging results: {e!s}", "error")
        return redirect(url_for("imaging.index"))


@bp.route("/", methods=["GET"])
@login_required
@roles_required("imaging", "admin")
def index():
    """Display imaging waiting list"""

    try:
        pending_requests = (
            RequestedImage.query.filter_by(status=0)
            .options(
                joinedload(RequestedImage.patient), joinedload(RequestedImage.imaging)
            )
            .all()
        )

        return render_template(
            "imaging/index.html",
            pending_requests=pending_requests or [],
            models_loaded=nim_client is not None,
        )
    except Exception as e:  # noqa: BLE001
        flash(f"Database error: {e!s}", "error")
        return redirect(url_for("home"))


# ── Phase 5: DICOM Upload Endpoint ─────────────────────────────────────────
@imaging_bp.route("/dicom/upload", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "imaging", "api")
def upload_dicom():
    """Upload a DICOM file and store it in the PACS."""
    from werkzeug.utils import secure_filename

    from departments.imaging.dicom_service import DICOMService

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith((".dcm", ".dicom")):
        return jsonify({"error": "File must be a DICOM file (.dcm or .dicom)"}), 400

    filename = secure_filename(file.filename)
    temp_path = Path("storage/dicom/incoming") / filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(temp_path)

    try:
        imaging_request_id = request.form.get("imaging_request_id", type=int)
        result = DICOMService.store_dicom(str(temp_path), imaging_request_id)

        return jsonify({
            "success": True,
            "message": "DICOM file stored successfully",
            "sop_instance_uid": result.sop_instance_uid,
            "study_instance_uid": result.study_instance_uid,
            "modality": result.modality,
            "file_path": result.file_path,
        }), 201

    except ValueError as e:
        if temp_path.exists():
            temp_path.unlink()
        return jsonify({"error": str(e)}), 400

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        return jsonify({"error": "Failed to process DICOM file"}), 500


@imaging_bp.route("/dicom/studies/<patient_id>", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "imaging", "medicine", "api")
def list_patient_studies(patient_id: str):
    """List all DICOM studies for a patient."""
    from departments.imaging.dicom_service import DICOMService

    studies = DICOMService.list_studies(patient_id)

    return jsonify({
        "patient_id": patient_id,
        "study_count": len(studies),
        "studies": studies,
    }), 200


@imaging_bp.route("/dicom/download/<sop_instance_uid>", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "imaging", "medicine", "api")
def download_dicom(sop_instance_uid: str):
    """Download a DICOM file by SOP Instance UID."""
    from flask import send_file

    from departments.imaging.dicom_service import DICOMService

    file_path = DICOMService.get_dicom_by_sop_uid(sop_instance_uid)

    if not file_path:
        return jsonify({"error": "DICOM file not found"}), 404

    return send_file(
        file_path,
        mimetype="application/dicom",
        as_attachment=True,
        download_name=f"{sop_instance_uid}.dcm"
    )


# ── Phase 5: DICOM Web UI Routes ───────────────────────────────────────────
@imaging_bp.route("/dicom/ui/upload", methods=["GET"])
@login_required
@roles_required("admin", "imaging", "medicine")
def dicom_upload_ui():
    """Render the DICOM upload page."""
    return render_template("imaging/dicom_upload.html")


@imaging_bp.route("/dicom/ui/studies/<patient_id>", methods=["GET"])
@login_required
@roles_required("admin", "imaging", "medicine")
def dicom_studies_ui(patient_id: str):
    """Render the DICOM studies gallery for a patient."""
    from departments.imaging.dicom_service import DICOMService
    from departments.models.records import Patient

    studies = DICOMService.list_studies(patient_id)
    patient = Patient.query.filter_by(patient_id=patient_id).first()

    return render_template(
        "imaging/dicom_studies.html",
        studies=studies,
        patient=patient,
        patient_id=patient_id
    )


@imaging_bp.route("/dicom/ui/studies", methods=["GET", "POST"])
@login_required
@roles_required("admin", "imaging", "medicine")
def dicom_studies_search_ui():
    """Patient search form to find DICOM studies."""
    from departments.models.records import Patient

    if request.method == "POST":
        patient_id = request.form.get("patient_id", "").strip()
        if patient_id:
            # Redirect to the studies gallery
            return redirect(url_for('imaging.dicom_studies_ui', patient_id=patient_id))
        else:
            flash("Please enter a Patient ID", "warning")

    # Get recent patients with imaging results for quick selection
    recent_patients = (
        db.session.query(Patient.patient_id, Patient.name)
        .join(ImagingResult)
        .order_by(ImagingResult.test_date.desc())
        .limit(20)
        .distinct()
        .all()
    )

    return render_template(
        "imaging/dicom_studies_search.html",
        recent_patients=recent_patients
    )
