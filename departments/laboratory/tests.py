import logging
from decimal import Decimal, InvalidOperation

from flask import abort, flash, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from departments.models.laboratory import LabResult, LabResultTemplate
from departments.models.medicine import LabTest, RequestedLab
from departments.rbac import has_any_role, roles_required
from extensions import db

from . import bp  # Import the blueprint

logger = logging.getLogger(__name__)


# Display the lab waiting list


@bp.route("/")
@login_required
@roles_required("laboratory", "admin")
def index():
    """Displays the laboratory waiting list using the unified patient-flow queue."""
    from departments.shared import queue_service

    try:
        # Unified patient flow queue: patients whose Encounter.stage is AWAITING_LAB
        pending_requests = queue_service.queue_for("laboratory")

        # Legacy fallback: also fetch raw RequestedLab rows for templates that
        # still reference .lab_test / .patient directly on the RequestedLab ORM.
        pending_lab_requests = (
            RequestedLab.query.filter_by(status=0)
            .options(
                joinedload(RequestedLab.patient),
                joinedload(RequestedLab.lab_test),
            )
            .all()
        )

        if not pending_lab_requests and not pending_requests:
            flash("No pending lab test requests at the moment.", "info")

        return render_template(
            "laboratory/index.html",
            pending_requests=pending_requests,
            pending_lab_requests=pending_lab_requests,
        )

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("Error in laboratory.index: %s", e)
        return redirect(url_for("laboratory.index"))


# display available lab tests
@bp.route("/lab_tests", methods=["GET"])
@login_required
@roles_required("laboratory", "admin")
def lab_tests():
    """Displays a list of available lab tests."""

    try:
        # Fetch all lab tests from the database
        lab_tests = LabTest.query.all()

        # Render the lab_tests.html template with the fetched data
        return render_template("laboratory/lab_tests.html", lab_tests=lab_tests)

    except SQLAlchemyError:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.lab_tests")
        return redirect(url_for("laboratory.index"))


# edit labtest
@bp.route("/edit_lab_test/<int:test_id>", methods=["GET", "POST"])
@login_required
@roles_required("laboratory", "admin", "medicine")
def edit_lab_test(test_id):
    """Handles editing a specific lab test."""

    try:
        # Fetch the lab test by ID
        lab_test = LabTest.query.get_or_404(test_id)

        # Fetch all associated result templates
        result_templates = LabResultTemplate.query.filter_by(test_id=test_id).all()

        if request.method == "POST":
            # P2-17 Segregation of duties: only admin/lab roles may change price.
            new_cost_raw = request.form.get("cost", "")
            if new_cost_raw and str(new_cost_raw) != str(lab_test.cost):
                if not has_any_role("laboratory", "admin"):
                    abort(403, description="Only laboratory/admin roles may change test pricing.")
                try:
                    lab_test.cost = Decimal(new_cost_raw).quantize(Decimal("0.01"))
                except (InvalidOperation, ValueError):
                    flash("Invalid cost value.", "error")
                    return redirect(url_for("laboratory.edit_lab_test", test_id=test_id))

            # Update lab test details
            lab_test.test_name = request.form.get("test_name", lab_test.test_name)
            lab_test.description = request.form.get("description", lab_test.description)
            lab_test.loinc_code = request.form.get("loinc_code", lab_test.loinc_code)

            # Process updated parameters
            updated_parameters = {}
            for key, value in request.form.items():
                if key.startswith("parameter_name_"):
                    param_id = key.split("_")[-1]
                    parameter_name = value.strip()
                    normal_range_low = request.form.get(
                        f"normal_range_low_{param_id}", ""
                    ).strip()
                    normal_range_high = request.form.get(
                        f"normal_range_high_{param_id}", ""
                    ).strip()
                    unit = request.form.get(f"unit_{param_id}", "").strip()

                    # Store updated parameter data only if parameter_name is provided
                    if parameter_name:  # Ensure parameter_name is not empty
                        updated_parameters[param_id] = {
                            "parameter_name": parameter_name,
                            "normal_range_low": float(normal_range_low)
                            if normal_range_low
                            else None,
                            "normal_range_high": float(normal_range_high)
                            if normal_range_high
                            else None,
                            "unit": unit,
                        }

            # Update existing result templates or delete if empty
            for template in result_templates:
                param_data = updated_parameters.get(str(template.id), {})
                if param_data.get("parameter_name"):  # If parameter name is not empty
                    template.parameter_name = param_data["parameter_name"]
                    template.normal_range_low = param_data["normal_range_low"]
                    template.normal_range_high = param_data["normal_range_high"]
                    template.unit = param_data["unit"]
                else:
                    # P1-11: deleting a parameter template corrupts history —
                    # stored results are keyed by template id and view_lab_results
                    # iterates only CURRENT templates, so deletion silently drops
                    # the parameter from every historical result. Block deletion
                    # while any result for this test exists.
                    has_results = LabResult.query.filter_by(
                        lab_test_id=test_id
                    ).first() is not None
                    if has_results:
                        flash(
                            f"Cannot delete parameter '{template.parameter_name}': "
                            "results for this test already exist and reference it. "
                            "Retain the parameter for historical integrity.",
                            "error",
                        )
                        return redirect(
                            url_for("laboratory.edit_lab_test", test_id=test_id)
                        )
                    db.session.delete(template)

            # Add new parameters (if any)
            new_param_count = int(request.form.get("new_param_count", 0))
            for i in range(1, new_param_count + 1):
                new_parameter_name = request.form.get(f"new_parameter_name_{i}").strip()
                new_normal_range_low = request.form.get(
                    f"new_normal_range_low_{i}", ""
                ).strip()
                new_normal_range_high = request.form.get(
                    f"new_normal_range_high_{i}", ""
                ).strip()
                new_unit = request.form.get(f"new_unit_{i}", "").strip()

                if new_parameter_name:  # Add only if parameter_name is not empty
                    new_template = LabResultTemplate(
                        test_id=test_id,
                        parameter_name=new_parameter_name,
                        normal_range_low=float(new_normal_range_low)
                        if new_normal_range_low
                        else None,
                        normal_range_high=float(new_normal_range_high)
                        if new_normal_range_high
                        else None,
                        unit=new_unit,
                    )
                    db.session.add(new_template)

            # Commit changes to the database
            db.session.commit()

            flash("Lab test updated successfully!", "success")
            return redirect(url_for("laboratory.view_lab_test", test_id=test_id))

        # Render the edit form on GET request
        return render_template(
            "laboratory/edit_lab_test.html",
            lab_test=lab_test,
            result_templates=result_templates,
        )

    except ValueError as ve:
        # Handle conversion errors (e.g., empty strings or invalid floats)
        flash(f"Invalid input: {ve}", "error")
        logger.warning("Invalid input in laboratory.edit_lab_test: %s", ve)
        return redirect(url_for("laboratory.edit_lab_test", test_id=test_id))

    except (SQLAlchemyError, ValueError, KeyError):
        flash("Something went wrong. Please try again.", "error")
        db.session.rollback()  # Rollback changes in case of error
        logger.exception("Error in laboratory.edit_lab_test")
        return redirect(url_for("laboratory.lab_tests"))


# delete test
@bp.route("/delete_lab_test/<test_id>", methods=["POST"])
@login_required
@roles_required("laboratory", "admin")
def delete_lab_test(test_id):
    """Handles deleting a lab test. Blocked when the test is in clinical use."""

    try:
        # Fetch the lab test by ID
        lab_test = LabTest.query.get_or_404(test_id)

        # P1-12: hard-deleting a test in use breaks FK integrity on every
        # referencing row — block with a clear message instead of a generic
        # "Something went wrong".
        requested_count = RequestedLab.query.filter_by(
            lab_test_id=lab_test.id
        ).count()
        result_count = LabResult.query.filter_by(lab_test_id=lab_test.id).count()
        if requested_count or result_count:
            flash(
                f"Cannot delete '{lab_test.test_name}': it is referenced by "
                f"{requested_count} request(s) and {result_count} result(s). "
                "Tests with clinical history cannot be deleted.",
                "error",
            )
            return redirect(url_for("laboratory.lab_tests"))

        # Delete the lab test
        db.session.delete(lab_test)
        db.session.commit()

        flash("Lab test deleted successfully!", "success")
        return redirect(url_for("laboratory.lab_tests"))

    except SQLAlchemyError:
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.delete_lab_test")
        return redirect(url_for("laboratory.lab_tests"))


# add lab test
@bp.route("/add_lab_test", methods=["GET", "POST"])
@login_required
@roles_required("laboratory", "admin")
def add_lab_test():
    """Handles adding a new lab test."""

    try:
        if request.method == "POST":
            # Extract form data
            test_name = (request.form.get("test_name") or "").strip()
            cost = request.form.get("cost")
            description = request.form.get("description")
            loinc_code = request.form.get("loinc_code")

            # Validation
            if not all([test_name, cost]):
                flash("Test name and cost are required!", "error")
                return render_template("laboratory/add_lab_test.html")

            # P2-17: prevent duplicate catalog entries (same name or LOINC)
            existing = LabTest.query.filter(
                (LabTest.test_name == test_name)
                | (LabTest.loinc_code == loinc_code if loinc_code else False)
            ).first()
            if existing:
                flash(
                    f"A test named '{existing.test_name}' already exists"
                    + (
                        f" (LOINC {existing.loinc_code})."
                        if existing.loinc_code
                        else "."
                    ),
                    "error",
                )
                return render_template("laboratory/add_lab_test.html")

            # P2-17: Numeric column — parse as Decimal, never float
            try:
                cost_decimal = Decimal(cost).quantize(Decimal("0.01"))
                if cost_decimal < 0:
                    raise InvalidOperation
            except (InvalidOperation, ValueError):
                flash("Invalid cost value.", "error")
                return render_template("laboratory/add_lab_test.html")

            # Create a new lab test
            new_lab_test = LabTest(
                test_name=test_name,
                cost=cost_decimal,
                description=description,
                loinc_code=loinc_code if loinc_code else None,
            )
            db.session.add(new_lab_test)
            db.session.commit()

            flash("Lab test added successfully!", "success")
            return redirect(url_for("laboratory.lab_tests"))

        # Render the add form on GET request
        return render_template("laboratory/add_lab_test.html")

    except (SQLAlchemyError, ValueError):
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.add_lab_test")
        return redirect(url_for("laboratory.lab_tests"))


@bp.route("/view_lab_test/<int:test_id>", methods=["GET"])
@login_required
@roles_required("laboratory", "admin", "medicine")
def view_lab_test(test_id):
    """Displays detailed information about a specific lab test."""

    try:
        lab_test = LabTest.query.get_or_404(test_id)
        result_templates = LabResultTemplate.query.filter_by(test_id=test_id).all()

        return render_template(
            "laboratory/view_lab_test.html",
            lab_test=lab_test,
            test_name=lab_test.test_name,
            description=lab_test.description,
            parameters=result_templates,
        )

    except SQLAlchemyError:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.view_lab_test")
        return redirect(url_for("laboratory.lab_tests"))


# process lab results


@bp.route("/dashboard")
@login_required
@roles_required("laboratory", "admin")
def dashboard():
    """Displays key lab statistics and trends."""

    total_tests = LabResult.query.count()
    pending_tests = RequestedLab.query.filter_by(status=0).count()

    # P2-13: the old metric string-matched '%abnormal%' inside the raw JSON
    # result blob — it only matched if someone typed the word "abnormal".
    # Use the panic_status column written by the threshold engine.
    abnormal_tests = LabResult.query.filter(
        LabResult.panic_status.in_(["ABNORMAL", "PANIC_CRITICAL"])
    ).count()
    panic_tests = LabResult.query.filter_by(panic_status="PANIC_CRITICAL").count()

    # P2-13: turnaround time (request -> result) over the last 30 days.
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    tat_rows = (
        db.session.query(RequestedLab.date_requested, LabResult.test_date)
        .join(
            LabResult,
            LabResult.result_id == RequestedLab.result_id,
        )
        .filter(
            RequestedLab.status == 1,
            RequestedLab.date_requested.isnot(None),
            RequestedLab.date_requested >= cutoff,
        )
        .all()
    )
    tat_hours = [
        (result_at - requested_at).total_seconds() / 3600.0
        for requested_at, result_at in tat_rows
        if result_at and result_at >= requested_at
    ]
    avg_tat_hours = round(sum(tat_hours) / len(tat_hours), 1) if tat_hours else None

    return render_template(
        "laboratory/dashboard.html",
        total_tests=total_tests,
        pending_tests=pending_tests,
        abnormal_tests=abnormal_tests,
        panic_tests=panic_tests,
        avg_tat_hours=avg_tat_hours,
    )
