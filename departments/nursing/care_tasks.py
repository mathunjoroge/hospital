import logging
from datetime import datetime, timezone

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.orm import joinedload

from departments.models.admin import Log
from departments.models.medicine import AdmittedPatient, Ward
from departments.models.nursing import MedicationAdmin, NursingCareTask, NursingNote
from departments.models.records import Patient
from departments.rbac import roles_required
from extensions import db

from . import bp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@bp.route("/search_patients", methods=["GET"])
@login_required  # ✅ Allow any logged-in user
def search_patients():
    """Search patients by name or patient ID without predefined boundaries."""
    try:
        query = request.args.get("q", "").strip()  # Ensure query is stripped

        if not query:
            return jsonify([])  # Return empty list if no query provided

        # Perform case-insensitive search for patients by name or ID
        results = (
            Patient.query.filter(
                (Patient.name.ilike(f"%{query}%"))
                | (Patient.patient_id.ilike(f"%{query}%"))
            )
            .limit(10)
            .all()
        )  # ✅ Limit to 10 results to optimize performance

        # Prepare response for Select2
        patient_list = [
            {
                "id": patient.patient_id,  # Use patient_id as the ID
                "text": f"{patient.name} ({patient.patient_id})",  # Show name + ID
            }
            for patient in results
        ]

        return jsonify(patient_list), 200

    except Exception as e:  # noqa: BLE001
        print(f"Debug: Error in search_patients: {e}")
        return jsonify({"error": "Failed to fetch patient data."}), 500


# search ward
@bp.route("/search_wards", methods=["GET"])
@login_required
def search_wards():
    """Fetch available wards dynamically for Select2."""
    query = request.args.get("q", "")

    wards = Ward.query.filter(Ward.name.ilike(f"%{query}%")).all()

    return jsonify(
        [
            {
                "id": ward.id,
                "text": f"{ward.name} ({ward.sex}) - Beds: {ward.available_beds()}",
            }
            for ward in wards
        ]
    )


@bp.route("/care_tasks", methods=["GET", "POST"])
@login_required
@roles_required("nursing", "admin")
def care_tasks():
    """Manage nursing care tasks."""

    if request.method == "POST":
        try:
            patient_id = request.form["patient_id"]
            task_description = request.form["task_description"]
            status = request.form.get("status", "Pending")
            new_task = NursingCareTask(
                patient_id=patient_id,
                nurse_id=current_user.id,
                task_description=task_description,
                status=status,
            )
            db.session.add(new_task)
            db.session.commit()
            logger.info(
                f"Nurse {current_user.id} added care task for patient {patient_id}"
            )
            db.session.add(
                Log(
                    level="INFO",
                    message=f"Nurse {current_user.username} (ID: {current_user.id}) added care task for patient {patient_id}",
                    user_id=current_user.id,
                    source="nursing",
                )
            )
            db.session.commit()
            flash("Care task added successfully.", "success")
        except Exception as e:
            db.session.rollback()
            flash("Something went wrong. Please try again.", "error")
            logger.exception("Error in nursing.care_tasks: ")
            db.session.add(
                Log(
                    level="ERROR",
                    message=f"Error adding care task: {e!s}",
                    user_id=current_user.id,
                    source="nursing",
                )
            )
            db.session.commit()

    try:
        tasks = (
            NursingCareTask.query.filter_by(nurse_id=current_user.id)
            .order_by(NursingCareTask.created_at.desc())
            .all()
        )
        logger.info(f"Nurse {current_user.id} viewed care tasks")
        db.session.add(
            Log(
                level="INFO",
                message=f"Nurse {current_user.username} (ID: {current_user.id}) viewed care tasks",
                user_id=current_user.id,
                source="nursing",
            )
        )
        db.session.commit()
        return render_template("nursing/care_tasks.html", tasks=tasks)
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in nursing.care_tasks: ")
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error loading care tasks: {e!s}",
                user_id=current_user.id,
                source="nursing",
            )
        )
        db.session.commit()
        return redirect(url_for("home"))


@bp.route("/care_summary", methods=["GET"])
@login_required
@roles_required("nursing", "admin")
def care_summary():
    """View a summary of nursing care for patients."""

    try:
        notes = (
            NursingNote.query.filter_by(nurse_id=current_user.id)
            .order_by(NursingNote.timestamp.desc())
            .all()
        )
        tasks = (
            NursingCareTask.query.filter_by(nurse_id=current_user.id)
            .order_by(NursingCareTask.created_at.desc())
            .all()
        )
        logger.info(f"Nurse {current_user.id} viewed care summary")
        db.session.add(
            Log(
                level="INFO",
                message=f"Nurse {current_user.username} (ID: {current_user.id}) viewed care summary",
                user_id=current_user.id,
                source="nursing",
            )
        )
        db.session.commit()
        return render_template("nursing/care_summary.html", notes=notes, tasks=tasks)
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in nursing.care_summary: ")
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error loading care summary: {e!s}",
                user_id=current_user.id,
                source="nursing",
            )
        )
        db.session.commit()
        return redirect(url_for("home"))


@bp.route("/medication_admin", methods=["GET", "POST"])
@login_required
@roles_required("nursing", "admin", "medicine")
def medication_admin():
    if request.method == "POST":
        try:
            patient_id = request.form.get("patient_id")
            medication = request.form.get("medication")
            dosage = request.form.get("dosage")
            if not patient_id or not medication or not dosage:
                flash("Patient ID, medication, and dosage are required.", "error")
                return redirect(url_for("nursing.medication_admin"))

            new_medication = MedicationAdmin(
                patient_id=patient_id,
                medication=medication,
                dosage=dosage,
                time_administered=datetime.now(timezone.utc),
                recorded_by=current_user.id,
            )
            db.session.add(new_medication)
            db.session.commit()
            flash("Medication administration recorded.", "success")
            return redirect(url_for("nursing.medication_admin"))
        except Exception as e:  # noqa: BLE001
            db.session.rollback()
            flash(f"Error recording medication: {e!s}", "error")
            return redirect(url_for("nursing.medication_admin"))

    return render_template("nursing/medication_admin.html")


@bp.route("/mark_task_completed/<int:task_id>")
@login_required
@roles_required("nursing")
def mark_task_completed(task_id):
    patient_id = request.args.get("patient_id", "").strip()
    if not patient_id:
        flash("Patient ID is required.", "error")
        return redirect(url_for("nursing.index"))

    try:
        task = NursingCareTask.query.filter_by(
            id=task_id, nurse_id=current_user.id
        ).first()
        if not task:
            flash(
                "Task not found or you do not have permission to mark it as completed.",
                "error",
            )
            return redirect(url_for("nursing.patient_dashboard", patient_id=patient_id))

        task.status = "Completed"
        task.completed_at = datetime.now(timezone.utc)
        db.session.commit()
        flash("Task marked as completed.", "success")
        return redirect(url_for("nursing.patient_dashboard", patient_id=patient_id))
    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash(f"Error marking task as completed: {e!s}", "error")
        return redirect(url_for("nursing.patient_dashboard", patient_id=patient_id))


# ─────────────────────────────────────────────
# WARD INPATIENTS MONITORING
# ─────────────────────────────────────────────
@bp.route("/ward-patients")
@login_required
@roles_required("nursing", "admin")
def ward_patients():
    admissions = (
        AdmittedPatient.query.filter_by(discharged_on=None)
        .options(joinedload(AdmittedPatient.patient), joinedload(AdmittedPatient.ward))
        .order_by(AdmittedPatient.admitted_on.desc())
        .all()
    )

    return render_template("nursing/ward_patients.html", admissions=admissions)


# ─────────────────────────────────────────────
# VISUAL VITALS TREND CHART
# ─────────────────────────────────────────────
