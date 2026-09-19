import os
import uuid
from datetime import datetime
from typing import Any

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required
from psycopg2.extras import RealDictCursor

from departments.models.medicine import (
    Imaging,
    LabTest,
    RequestedImage,
    RequestedLab,
    SOAPNote,
    UnmatchedImagingRequest,
)
from departments.models.records import Patient
from departments.nlp.chatbot import UniversalClinicalSummarizer
from departments.nlp.logging_setup import get_logger
from departments.rbac import roles_required
from departments.shared.encounter_utils import active_encounter
from extensions import db, socketio

from . import bp

logger = get_logger()

# Instantiate the summarizer for use in chatbot_interface


gemini_api_key = os.environ.get("GEMINI_API_KEY")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
Summarizer = UniversalClinicalSummarizer(
    gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key
)


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "txt", "csv", "docx"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size


@bp.route("/request_lab_tests/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def request_lab_tests(patient_id):
    """Handles lab test requests (OPD waiting-list and admitted IPD patients)."""
    try:
        dept = request.args.get("dept")  # ✅ Capture dept from query string

        # Resolve the patient from the Patient master directly. The old
        # waiting-list-only lookup meant admitted (IPD) patients could never
        # have labs ordered from the ward.
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f"Patient with ID {patient_id} not found!", "error")
            return redirect(url_for("medicine.index"))

        lab_tests = LabTest.query.all()

        if request.method == "POST":
            lab_test_ids = request.form.getlist("lab_tests[]")
            if not lab_test_ids:
                flash("No lab tests selected!", "error")
                return render_template(
                    "medicine/request_lab_tests.html",
                    patient=patient,
                    lab_tests=lab_tests,
                    dept=dept,
                )

            descriptions = {}
            for key, value in request.form.items():
                if key.startswith("descriptions["):
                    lab_id = key.split("[")[1].split("]")[0]
                    descriptions[lab_id] = value.strip() if value else ""

            encounter = active_encounter(patient_id)

            # Duplicate guard: skip tests already requested and still pending
            # for this patient (double-click / refresh resubmits).
            pending_test_ids = {
                row.lab_test_id
                for row in RequestedLab.query.filter_by(
                    patient_id=patient_id, status=0
                ).all()
            }

            for lab_test_id in lab_test_ids:
                description = descriptions.get(str(lab_test_id), "").strip()

                if len(description) > 500:
                    flash(
                        f"Description for lab test ID {lab_test_id} exceeds 500 characters.",
                        "error",
                    )
                    return render_template(
                        "medicine/request_lab_tests.html",
                        patient=patient,
                        lab_tests=lab_tests,
                        dept=dept,
                    )

                if int(lab_test_id) in pending_test_ids:
                    continue  # already requested and pending

                result_id = str(uuid.uuid4())

                new_lab_request = RequestedLab(
                    patient_id=patient_id,
                    encounter_id=encounter.id if encounter else None,
                    lab_test_id=lab_test_id,
                    result_id=result_id,
                    description=description or None,
                )
                db.session.add(new_lab_request)

            db.session.commit()
            flash("Lab tests requested successfully!", "success")

            # ✅ Advance encounter stage: lab ordered → AWAITING_LAB
            if encounter and encounter.stage in ("WAITING_DOCTOR", "IN_CONSULTATION"):
                encounter.stage = "AWAITING_LAB"
                db.session.add(encounter)
                db.session.commit()

            # ✅ Redirect accordingly
            if dept == "1":
                return redirect(url_for("medicine.ward_rounds"))
            else:
                return redirect(url_for("medicine.soap_notes", patient_id=patient_id))

        # GET request
        return render_template(
            "medicine/request_lab_tests.html",
            patient=patient,
            lab_tests=lab_tests,
            dept=dept,
        )

    except Exception:
        db.session.rollback()
        logger.exception("Error in medicine.request_lab_tests: ")
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("medicine.soap_notes", patient_id=patient_id))


@bp.route("/request_imaging/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def request_imaging(patient_id):
    """Handles imaging requests (OPD waiting-list and admitted IPD patients)."""
    try:
        dept = request.args.get("dept")  # ✅ Capture dept from query string

        # Resolve from Patient master directly (see request_lab_tests note).
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f"Patient with ID {patient_id} not found!", "error")
            return redirect(url_for("medicine.index"))

        soap_notes = (
            SOAPNote.query.filter_by(patient_id=patient_id)
            .order_by(SOAPNote.created_at.desc())
            .first()
        )
        imaging_types = Imaging.query.all()

        if request.method == "POST":
            imaging_ids = request.form.getlist("imaging_types[]")

            if not imaging_ids:
                flash("No imaging types selected!", "error")
                return render_template(
                    "medicine/request_imaging.html",
                    patient=patient,
                    soap_notes=soap_notes,
                    imaging_types=imaging_types,
                    dept=dept,
                )

            descriptions = {}
            for key, value in request.form.items():
                if key.startswith("descriptions["):
                    imaging_id = key.split("[")[1].split("]")[0]
                    descriptions[imaging_id] = value.strip() if value else ""

            encounter = active_encounter(patient_id)

            # Duplicate guard for pending imaging requests.
            pending_imaging_ids = {
                row.imaging_id
                for row in RequestedImage.query.filter_by(
                    patient_id=patient_id, status=0
                ).all()
            }

            for imaging_id in imaging_ids:
                description = descriptions.get(str(imaging_id), "").strip()

                if len(description) > 500:
                    flash(
                        f"Description for imaging ID {imaging_id} exceeds 500 characters.",
                        "error",
                    )
                    return render_template(
                        "medicine/request_imaging.html",
                        patient=patient,
                        soap_notes=soap_notes,
                        imaging_types=imaging_types,
                        dept=dept,
                    )

                if int(imaging_id) in pending_imaging_ids:
                    continue  # already requested and pending

                result_id = str(uuid.uuid4())
                new_image_request = RequestedImage(
                    patient_id=patient_id,
                    encounter_id=encounter.id if encounter else None,
                    imaging_id=imaging_id,
                    result_id=result_id,
                    description=description or None,
                )
                db.session.add(new_image_request)

            db.session.commit()
            flash("Imaging requested successfully!", "success")

            # ✅ Advance encounter stage: imaging ordered → AWAITING_IMAGING
            if encounter and encounter.stage in (
                "WAITING_DOCTOR",
                "IN_CONSULTATION",
                "AWAITING_LAB",
            ):
                encounter.stage = "AWAITING_IMAGING"
                db.session.add(encounter)
                db.session.commit()

            # ✅ Redirect based on dept
            if dept == "1":
                return redirect(url_for("medicine.ward_rounds"))
            else:
                return redirect(url_for("medicine.soap_notes", patient_id=patient_id))

        # GET: Render form
        return render_template(
            "medicine/request_imaging.html",
            patient=patient,
            soap_notes=soap_notes,
            imaging_types=imaging_types,
            dept=dept,
        )

    except Exception:
        db.session.rollback()
        logger.exception("Error in medicine.request_imaging: ")
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("medicine.soap_notes", patient_id=patient_id))


@bp.route("/unmatched_imaging", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def unmatched_imaging():
    """Medicine panel to match unmatched imaging requests."""
    # Handle form submission (matching requests)
    if request.method == "POST":
        unmatched_id = request.form.get("unmatched_id")
        imaging_id = request.form.get("imaging_id")

        if unmatched_id and imaging_id:
            unmatched_request = db.session.get(UnmatchedImagingRequest, unmatched_id)
            if unmatched_request:
                # Move to requested_images table. UnmatchedImagingRequest
                # doesn't carry its own encounter_id, so scope to whatever
                # encounter is active for the patient now (best effort).
                encounter = active_encounter(unmatched_request.patient_id)
                requested_imaging = RequestedImage(
                    patient_id=unmatched_request.patient_id,
                    encounter_id=encounter.id if encounter else None,
                    imaging_id=imaging_id,
                    description=unmatched_request.description,
                )
                db.session.add(requested_imaging)

                # Remove from unmatched list
                db.session.delete(unmatched_request)
                db.session.commit()
                flash("Imaging request successfully matched!", "success")
            else:
                flash("Invalid request!", "error")

    # Get filtering parameters (validate dates: raw strings passed straight
    # to a DateTime comparison raised a 500 on invalid input)
    patient_name = request.args.get("patient_name", "").strip()
    start_date_raw = request.args.get("start_date", "")
    end_date_raw = request.args.get("end_date", "")

    def _parse_date(raw, end_of_day=False):
        if not raw:
            return None
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d")
            if end_of_day:
                parsed = parsed.replace(hour=23, minute=59, second=59)
            return parsed
        except ValueError:
            flash("Invalid date filter — use YYYY-MM-DD.", "error")
            return None

    start_date = _parse_date(start_date_raw)
    end_date = _parse_date(end_date_raw, end_of_day=True)

    # Base query for unmatched imaging requests
    unmatched_requests = UnmatchedImagingRequest.query.join(Patient).order_by(
        UnmatchedImagingRequest.date_requested.desc()
    )

    # Apply filters if provided
    if patient_name:
        unmatched_requests = unmatched_requests.filter(
            Patient.name.ilike(f"%{patient_name}%")
        )
    if start_date:
        unmatched_requests = unmatched_requests.filter(
            UnmatchedImagingRequest.date_requested >= start_date
        )
    if end_date:
        unmatched_requests = unmatched_requests.filter(
            UnmatchedImagingRequest.date_requested <= end_date
        )

    unmatched_requests = unmatched_requests.all()
    imaging_options = Imaging.query.all()

    return render_template(
        "medicine/unmatched_imaging.html",
        unmatched_requests=unmatched_requests,
        imaging_options=imaging_options,
        patient_name=patient_name,
        start_date=start_date_raw,
        end_date=end_date_raw,
    )


@bp.route("/unmatched_imaging/notify", methods=["GET", "POST"])
@login_required
def notify_admin():
    """Notify medicine users by updating badge count via SocketIO."""
    count = UnmatchedImagingRequest.query.count()
    socketio.emit(
        "update_badge", {"count": count}, namespace="/medicine"
    )  # Updated namespace
    return "", 204  # Return a success response without content


def get_unmatched_count():
    """Get the number of unmatched imaging requests."""
    return UnmatchedImagingRequest.query.count()


@bp.context_processor
def inject_unmatched_count():
    """Inject the unmatched count into the template context."""
    return {"unmatched_count": get_unmatched_count()}


from departments.shared.drugcentral import (
    get_drugcentral_connection as get_db_connection,
)


def fetch_drugs_data(
    search_query: str | None = None, category: str | None = None
) -> list[dict[str, Any]]:
    """Fetch distinct product data with optional search by generic name, brand name, or therapeutic category."""
    results: list[dict[str, Any]] = []
    seen = set()

    # Category keyword mapping for quick clinical pills
    CATEGORY_KEYWORDS = {
        "antibiotics": ["cillin", "cef", "floxacin", "mycin", "cycline", "sulfa", "azole", "penem"],
        "cardiovascular": ["lol", "pril", "sartan", "dipine", "statin", "furosemide", "digoxin", "nitrate"],
        "analgesics": ["paracetamol", "ibuprofen", "morphine", "tramadol", "codeine", "diclofenac", "naproxen", "aspirin"],
        "oncology": ["cisplatin", "doxorubicin", "paclitaxel", "fluorouracil", "methotrexate", "tamoxifen", "cyclophosphamide"],
        "antidiabetics": ["metformin", "glipizide", "insulin", "empagliflozin", "sitagliptin", "gliclazide"],
        "respiratory": ["salbutamol", "budesonide", "montelukast", "theophylline", "ipratropium", "fluticasone"],
        "cns_psychiatry": ["diazepam", "lorazepam", "haloperidol", "sertraline", "fluoxetine", "olanzapine", "carbamazepine"],
        "gastrointestinal": ["omeprazole", "pantoprazole", "ranitidine", "metoclopramide", "loperamide", "ondansetron"],
    }

    try:
        with get_db_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            base_query = """
                SELECT DISTINCT generic_name, product_name, route, form
                FROM product
            """

            where_clauses = []
            params = []

            if search_query:
                search_param = f"%{search_query}%"
                where_clauses.append("(generic_name ILIKE %s OR product_name ILIKE %s)")
                params.extend([search_param, search_param])

            if category and category.lower() in CATEGORY_KEYWORDS:
                cat_patterns = CATEGORY_KEYWORDS[category.lower()]
                cat_clause_parts = []
                for pat in cat_patterns:
                    cat_clause_parts.append("generic_name ILIKE %s OR product_name ILIKE %s")
                    params.extend([f"%{pat}%", f"%{pat}%"])
                if cat_clause_parts:
                    where_clauses.append(f"({' OR '.join(cat_clause_parts)})")

            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)

            base_query += " ORDER BY generic_name LIMIT 200"
            cur.execute(base_query, params)
            for row in cur.fetchall():
                item = dict(row)
                key = (
                    (item.get("generic_name") or "").lower(),
                    (item.get("product_name") or "").lower(),
                )
                if key not in seen:
                    seen.add(key)
                    results.append(item)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "DrugCentral connection unavailable (%s); using local hospital database.", exc
        )

    # Local hospital database medicines & pharmacy drugs fallback/supplement
    try:
        from departments.models.medicine import Medicine
        from departments.models.pharmacy import Drug as PharmDrug

        query = Medicine.query
        if search_query:
            pattern = f"%{search_query}%"
            query = query.filter(
                (Medicine.generic_name.ilike(pattern))
                | (Medicine.brand_name.ilike(pattern))
            )
        for m in query.order_by(Medicine.generic_name).all():
            key = (m.generic_name.lower(), m.brand_name.lower())
            if key not in seen:
                seen.add(key)
                results.append(
                    {
                        "generic_name": m.generic_name,
                        "product_name": m.brand_name,
                        "route": "Oral / Systemic",
                        "form": m.dosage,
                    }
                )

        p_query = PharmDrug.query
        if search_query:
            pattern = f"%{search_query}%"
            p_query = p_query.filter(PharmDrug.generic_name.ilike(pattern))
        for pd in p_query.order_by(PharmDrug.generic_name).all():
            brand = pd.generic_name
            key = (pd.generic_name.lower(), brand.lower())
            if key not in seen:
                seen.add(key)
                form_str = (
                    f"{pd.dosage_form or ''} {pd.strength or ''}".strip()
                    or "Standard"
                )
                results.append(
                    {
                        "generic_name": pd.generic_name,
                        "product_name": brand,
                        "route": "Pharmacy Stock",
                        "form": form_str,
                    }
                )
    except Exception:  # noqa: BLE001
        logger.exception("Error fetching local hospital database medicines")

    return results
