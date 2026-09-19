import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from flask import Response, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from departments.models.laboratory import LabResult, LabResultTemplate
from departments.models.medicine import LabTest, RequestedLab
from departments.models.records import Patient
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

logger = logging.getLogger(__name__)

# Display the lab waiting list


@bp.route("/process_lab_request/<int:request_id>", methods=["GET", "POST"])
@login_required
@roles_required("laboratory", "admin")
def process_lab_request(request_id):
    """Handles processing a lab test request."""

    try:
        # Fetch the requested lab test by ID
        lab_request = RequestedLab.query.get_or_404(request_id)

        # P1-7 idempotency: a result already recorded for this request must not
        # be silently duplicated (double-click / retry protection).
        if lab_request.status == 1 and lab_request.result_id:
            existing = LabResult.query.filter_by(
                result_id=lab_request.result_id
            ).first()
            if existing:
                flash(
                    "This lab request has already been processed. Showing the recorded result.",
                    "info",
                )
                return redirect(
                    url_for("laboratory.view_lab_results", result_id=existing.id)
                )

        # Fetch the associated lab test details
        lab_test = LabTest.query.get_or_404(lab_request.lab_test_id)

        # Fetch all parameters (templates) for the lab test
        parameters = LabResultTemplate.query.filter_by(test_id=lab_test.id).all()

        if request.method == "POST":
            # Extract form data for lab_test_id[] and result[]
            lab_test_ids = request.form.getlist("lab_test_id[]")
            results = request.form.getlist("result[]")

            # Combine lab_test_ids and results into a dictionary
            results_dict = {
                param: result.strip()
                for param, result in zip(lab_test_ids, results)
                if result.strip()
            }

            # Validate that all required fields are provided
            if not results_dict:
                flash("At least one result must be entered!", "error")
                return render_template(
                    "laboratory/process_lab_request.html",
                    lab_request=lab_request,
                    lab_test=lab_test,
                    parameters=parameters,
                    result_id=str(uuid.uuid4()),
                )

            # P2-16: per-request UUID — no shared session key, so two tabs
            # processing different requests cannot clobber each other's id.
            result_id = str(uuid.uuid4())

            # P1-6: run every entered value through the panic-threshold engine
            # so web-entered results get the same safety evaluation as LIS ones.
            from departments.laboratory.panic_alerts import evaluate_panic_level

            overall_panic_status = "NORMAL"
            overall_panic_msgs = []
            for param_id, value in results_dict.items():
                param = db.session.get(LabResultTemplate, int(param_id))
                if not param:
                    continue
                try:
                    value_f = float(value)
                except ValueError:
                    continue  # non-numeric (qualitative) results are not scored
                status, msg = evaluate_panic_level(param.parameter_name, value_f)
                if status == "PANIC_CRITICAL":
                    overall_panic_status = "PANIC_CRITICAL"
                    overall_panic_msgs.append(msg)
                elif status == "ABNORMAL" and overall_panic_status == "NORMAL":
                    overall_panic_status = "ABNORMAL"
                    overall_panic_msgs.append(msg)

            panic_message = "\n".join(overall_panic_msgs) or None

            # Create the lab result record
            lab_result = LabResult(
                patient_id=lab_request.patient_id,
                lab_test_id=lab_test.id,
                test_date=datetime.now(timezone.utc),
                result_notes=request.form.get("result_notes", ""),  # Optional notes
                result=json.dumps(results_dict),  # Store the results as a JSON string
                result_id=result_id,  # Assign the unique result_id
                updated_by=current_user.id,  # Set the user who processed the result
                status="PENDING_VERIFICATION",
                panic_status=overall_panic_status,
                panic_message=panic_message,
            )
            db.session.add(lab_result)

            # Mark the request processed and link it to the result — all in ONE
            # commit so a partial failure can never leave a pending request
            # pointing at nothing (or vice versa).
            lab_request.status = 1
            lab_request.result_id = result_id
            db.session.add(lab_request)
            db.session.commit()

            # P1-8: advance any linked specimen into analysis/completed states.
            _advance_specimens_for_request(lab_request)

            # FIX 4: Advance encounter stage after lab completion
            from departments.shared.visit_closure import advance_after_completion

            advance_after_completion(lab_request.patient_id)

            # Trigger notification to patient that lab result is ready
            from departments.notifications.triggers import trigger_lab_result_ready

            trigger_lab_result_ready(lab_request)

            if overall_panic_status == "PANIC_CRITICAL":
                flash(
                    "Results saved — CRITICAL PANIC VALUE detected. Pathologist "
                    "verification and clinician notification required.",
                    "danger",
                )
            elif overall_panic_status == "ABNORMAL":
                flash("Results saved — abnormal value(s) flagged for review.", "warning")
            else:
                flash("Lab test results submitted successfully!", "success")
            return redirect(
                url_for("laboratory.view_lab_results", result_id=lab_result.id)
            )

        # Render the form on GET request
        return render_template(
            "laboratory/process_lab_request.html",
            lab_request=lab_request,
            lab_test=lab_test,
            parameters=parameters,
            result_id=str(uuid.uuid4()),
        )

    except (SQLAlchemyError, ValueError, KeyError, json.JSONDecodeError) as e:
        flash("Something went wrong. Please try again.", "error")
        db.session.rollback()  # Rollback changes in case of error
        logger.exception("Error in laboratory.process_lab_request")
        return redirect(url_for("laboratory.index"))


def _advance_specimens_for_request(lab_request):
    """
    P1-8: bridge the specimen-tracking workflow and the testing workflow.
    Un-collected specimens for this request are marked IN_ANALYSIS once a
    result is recorded; received ones advance to COMPLETED with a
    chain-of-custody entry, via LIMSService so the CoC log stays intact.
    """
    from departments.laboratory.lims_service import LIMSService

    for specimen in lab_request.specimens:
        if specimen.status == "ORDERED":
            LIMSService.update_specimen_status(
                specimen.id, "IN_ANALYSIS", user_id=current_user.id,
                notes="Result entry started",
            )
        elif specimen.status in ("RECEIVED", "COLLECTED"):
            LIMSService.update_specimen_status(
                specimen.id, "COMPLETED", user_id=current_user.id,
                notes=f"Result recorded ({lab_request.result_id})",
            )


# view lab results
@bp.route("/view_lab_results/<int:result_id>", methods=["GET"])
@login_required
@roles_required("laboratory", "admin")
def view_lab_results(result_id):
    """Displays lab test results in a structured format."""

    try:
        # Fetch the lab result by ID
        lab_result = LabResult.query.get_or_404(result_id)

        # Parse the result string back into a dictionary
        try:
            results_dict = json.loads(lab_result.result) if lab_result.result else {}
        except json.JSONDecodeError:
            flash("Something went wrong. Please try again.", "warning")
            results_dict = {}  # Fallback to an empty dictionary if parsing fails

        # Fetch the associated lab test details
        lab_test = LabTest.query.get_or_404(lab_result.lab_test_id)

        # Fetch all parameters (templates) for the lab test
        parameters = LabResultTemplate.query.filter_by(test_id=lab_test.id).all()

        # Match results with parameter IDs and prepare presentation data
        test_presentation = []
        for param in parameters:
            result_value = results_dict.get(str(param.id))  # Ensure key is a string

            try:
                # Convert result to float for comparisons
                result_value_float = (
                    float(result_value) if result_value is not None else None
                )
            except ValueError:
                result_value_float = None  # Handle invalid data

            # Determine the status
            if result_value_float is None:
                status = "Invalid Result"
            elif result_value_float < param.normal_range_low:
                status = "Low"
            elif result_value_float > param.normal_range_high:
                status = "High"
            else:
                status = "Normal"

            # Append to test_presentation
            test_presentation.append(
                {
                    "parameter_name": param.parameter_name,
                    "normal_range_low": param.normal_range_low,
                    "normal_range_high": param.normal_range_high,
                    "unit": param.unit,
                    "result": result_value
                    if result_value is not None
                    else "N/A",  # Handle missing results
                    "status": status,  # Set final status
                }
            )

        return render_template(
            "laboratory/view_lab_results.html",
            lab_test=lab_test,
            lab_result=lab_result,
            test_presentation=test_presentation,
        )

    except (SQLAlchemyError, ValueError, KeyError) as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.view_lab_results")
        return redirect(url_for("laboratory.index"))


@bp.route("/pending_lab_results")
@login_required
@roles_required("laboratory", "admin")
def pending_lab_results():
    """Displays pending lab test requests."""

    try:
        # Fetch all pending lab test requests
        pending_lab_requests = (
            RequestedLab.query.filter_by(status=0)
            .options(
                joinedload(RequestedLab.patient), joinedload(RequestedLab.lab_test)
            )
            .all()
        )

        return render_template(
            "laboratory/pending_lab_results.html",
            pending_lab_requests=pending_lab_requests,
        )

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.pending_lab_results")
        return redirect(url_for("laboratory.index"))


@bp.route("/processed_lab_results")
@login_required
@roles_required("laboratory", "admin")
def processed_lab_results():
    """Displays processed lab test results (paginated)."""

    try:
        page = request.args.get("page", 1, type=int)
        per_page = 50

        # Fetch processed lab results where updated_by is NOT NULL
        pagination = (
            db.session.query(
                LabResult.id,
                LabResult.patient_id,
                LabResult.lab_test_id,
                LabResult.test_date,
                LabResult.result_id,
                LabResult.result_notes,
                LabResult.updated_by,
                LabTest.test_name,
                Patient.name.label("patient_name"),
            )
            .join(LabTest, LabTest.id == LabResult.lab_test_id)
            .join(Patient, Patient.patient_id == LabResult.patient_id)
            .filter(LabResult.updated_by.isnot(None))
            .order_by(LabResult.test_date.desc())
            .paginate(page=page, per_page=per_page, error_out=False)
        )
        processed_lab_results = pagination.items

        # If no processed results exist, inform the user
        if not processed_lab_results:
            flash("No processed lab results available.", "info")

        return render_template(
            "laboratory/processed_lab_results.html",
            processed_lab_results=processed_lab_results,
            pagination=pagination,
        )

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.processed_lab_results")
        return redirect(url_for("laboratory.index"))


@bp.route("/generate_lab_report/<int:result_id>")
@login_required
@roles_required("laboratory", "admin")
def generate_lab_report(result_id):
    """Generates a PDF lab test report."""

    result = LabResult.query.get_or_404(result_id)

    response = Response(content_type="application/pdf")
    response.headers["Content-Disposition"] = (
        f"inline; filename=Lab_Report_{result_id}.pdf"
    )

    pdf = canvas.Canvas(response.stream, pagesize=letter)
    pdf.drawString(
        100,
        750,
        f"Lab Report for {result.patient.name} - {result.test_date.strftime('%Y-%m-%d')}",
    )
    pdf.drawString(100, 730, f"Test: {result.lab_test.test_name}")
    pdf.drawString(100, 710, f"Results: {result.result}")

    pdf.showPage()
    pdf.save()

    return response


@bp.route("/search-patient", methods=["GET", "POST"])
@login_required
@roles_required("laboratory", "admin")
def search_patient():
    """Search for a patient and fetch their lab history on the same page."""

    patients = []
    lab_results = []
    selected_patient = None  # Track which patient is selected

    if request.method == "POST":
        search_query = request.form.get("search_query")

        if search_query:
            # Search for patients by ID or Name (case insensitive)
            patients = Patient.query.filter(
                (Patient.patient_id.ilike(f"%{search_query}%"))
                | (Patient.name.ilike(f"%{search_query}%"))
            ).all()

            if patients:
                # Only fetch lab history if a patient is selected
                selected_patient_id = request.form.get("selected_patient_id")

                if selected_patient_id:
                    selected_patient = Patient.query.filter_by(
                        patient_id=selected_patient_id
                    ).first()

                    if selected_patient:
                        lab_results = (
                            db.session.query(
                                LabResult.id,
                                LabResult.test_date,
                                LabResult.result_id,
                                LabResult.result_notes,
                                LabTest.test_name,
                            )
                            .join(LabTest, LabResult.lab_test_id == LabTest.id)
                            .filter(LabResult.patient_id == selected_patient.patient_id)
                            .order_by(LabResult.test_date.desc())
                            .all()
                        )
            else:
                flash("No patients found with that name or ID.", "warning")

    return render_template(
        "laboratory/search_patient.html",
        patients=patients,
        lab_results=lab_results,
        selected_patient=selected_patient,
    )


# abnormal results
@bp.route("/abnormal_results")
@login_required
@roles_required("laboratory", "admin")
def abnormal_results():
    """
    Displays lab test results that are outside normal ranges.

    P1-10: queries the panic_status column written at result-entry time
    instead of scanning and JSON-parsing every LabResult in the database.
    Legacy rows predating the panic engine are still covered by a bounded
    90-day JSON re-scan fallback.
    """

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)

        # Primary source: panic evaluation persisted at entry time
        engine_flagged = (
            db.session.query(
                LabResult.id,
                LabResult.patient_id,
                LabResult.lab_test_id,
                LabResult.result,
                LabResult.panic_status,
                LabResult.panic_message,
                LabTest.test_name,
            )
            .join(LabTest, LabResult.lab_test_id == LabTest.id)
            .filter(
                LabResult.panic_status.in_(["PANIC_CRITICAL", "ABNORMAL"]),
                LabResult.test_date >= cutoff,
            )
            .order_by(LabResult.test_date.desc())
            .limit(200)
            .all()
        )

        # Legacy rows (no panic evaluation) within the same window
        legacy_rows = (
            db.session.query(
                LabResult.id,
                LabResult.patient_id,
                LabResult.lab_test_id,
                LabResult.result,
                LabTest.test_name,
            )
            .join(LabTest, LabResult.lab_test_id == LabTest.id)
            .filter(
                LabResult.panic_status == "NORMAL",  # default — never evaluated
                LabResult.updated_by.isnot(None),
                LabResult.test_date >= cutoff,
            )
            .order_by(LabResult.test_date.desc())
            .limit(500)
            .all()
        )

        flagged_results = []

        # Engine-flagged results: split the stored panic message per parameter
        for result in engine_flagged:
            abnormal_parameters = []
            try:
                result_data = json.loads(result.result) if result.result else {}
            except json.JSONDecodeError:
                result_data = {}
            for param_id, value in result_data.items():
                param = db.session.get(LabResultTemplate, int(param_id))
                if not param:
                    continue
                try:
                    value_f = float(value)
                except ValueError:
                    continue
                if (
                    value_f < param.normal_range_low
                    or value_f > param.normal_range_high
                ):
                    abnormal_parameters.append(
                        {
                            "parameter": param.parameter_name,
                            "value": value,
                            "normal_range": f"{param.normal_range_low} - {param.normal_range_high}",
                            "unit": param.unit,
                        }
                    )
            if abnormal_parameters:
                flagged_results.append(
                    {
                        "id": result.id,
                        "patient_id": result.patient_id,
                        "test_name": result.test_name,
                        "panic_status": result.panic_status,
                        "abnormal_parameters": abnormal_parameters,
                    }
                )
            else:
                # Message-based fallback (LIS single-value results)
                flagged_results.append(
                    {
                        "id": result.id,
                        "patient_id": result.patient_id,
                        "test_name": result.test_name,
                        "panic_status": result.panic_status,
                        "abnormal_parameters": [
                            {
                                "parameter": result.result or "Result",
                                "value": "",
                                "normal_range": "",
                                "unit": "",
                            }
                        ],
                    }
                )

        # Legacy fallback scan (bounded)
        for result in legacy_rows:
            try:
                result_data = json.loads(result.result) if result.result else {}
                abnormal_parameters = []

                for param_id, value in result_data.items():
                    param = db.session.get(LabResultTemplate, param_id)
                    if param and (
                        float(value) < param.normal_range_low
                        or float(value) > param.normal_range_high
                    ):
                        abnormal_parameters.append(
                            {
                                "parameter": param.parameter_name,
                                "value": value,
                                "normal_range": f"{param.normal_range_low} - {param.normal_range_high}",
                                "unit": param.unit,
                            }
                        )

                if abnormal_parameters:
                    flagged_results.append(
                        {
                            "id": result.id,
                            "patient_id": result.patient_id,
                            "test_name": result.test_name,
                            "panic_status": "ABNORMAL",
                            "abnormal_parameters": abnormal_parameters,
                        }
                    )

            except (ValueError, KeyError, json.JSONDecodeError):
                logger.debug("Skipping unparseable lab result %s", result.id)

        return render_template(
            "laboratory/abnormal_results.html", flagged_results=flagged_results
        )

    except SQLAlchemyError:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.abnormal_results")
        return redirect(url_for("laboratory.index"))


# verification queue
@bp.route("/verification_queue")
@login_required
@roles_required("laboratory", "admin", "medicine")
def verification_queue():
    """
    P1-6: web UI for the 2-tier verification workflow. Lists results entered
    by the LIS API or web entry that are still PENDING_VERIFICATION, so a
    pathologist/doctor can sign them off (via the LIS verify API).
    """
    pending = (
        db.session.query(
            LabResult.id,
            LabResult.result_id,
            LabResult.patient_id,
            LabResult.test_date,
            LabResult.panic_status,
            LabResult.panic_message,
            LabResult.updated_by,
            LabTest.test_name,
            Patient.name.label("patient_name"),
        )
        .join(LabTest, LabTest.id == LabResult.lab_test_id)
        .join(Patient, Patient.patient_id == LabResult.patient_id)
        .filter(LabResult.status == "PENDING_VERIFICATION")
        .order_by(LabResult.test_date.asc())  # oldest first — criticals age worst
        .limit(200)
        .all()
    )

    return render_template("laboratory/verification_queue.html", pending=pending)


# dashboard
