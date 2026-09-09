import logging
from datetime import datetime, timezone

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.orm import joinedload

from departments.appointments.engine import ScheduleEngine
from departments.models.admin import Log
from departments.models.nursing import Partogram, Vitals
from departments.models.records import Patient, PatientWaitingList
from departments.shared import queue_service
from departments.models.user import User
from departments.rbac import roles_required
from departments.shared.queue_constants import QueueStatus
from extensions import db

from . import bp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@bp.route("/")
@login_required
@roles_required("nursing", "medicine", "admin")
def index():
    """Display the nursing waiting list."""
    try:
        # Phase 3: read from Encounter.stage via QueueService.
        valid_waiting_list = queue_service.queue_for("nursing")

        return render_template("nursing/index.html", waiting_list=valid_waiting_list)
    except Exception as e:
        # Handle any unexpected errors
        flash("Something went wrong. Please try again.", "error")
        print(f"Error in nursing.index: {e}")  # Debugging
        return redirect(url_for("nursing.index"))
    # vitals


@bp.route("/vitals/<patient_id>", methods=["GET", "POST"])  # Removed <int:>
@login_required
@roles_required("nursing", "admin")
def vitals(patient_id):
    """Manage vital signs for a patient."""
    if request.method == "POST":
        try:
            vitals_data = Vitals(
                patient_id=patient_id,  # String patient_id
                nurse_id=current_user.id,
                temperature=float(request.form["temperature"])
                if request.form["temperature"]
                else None,
                pulse=int(request.form["pulse"]) if request.form["pulse"] else None,
                blood_pressure_systolic=int(request.form["blood_pressure_systolic"])
                if request.form["blood_pressure_systolic"]
                else None,
                blood_pressure_diastolic=int(request.form["blood_pressure_diastolic"])
                if request.form["blood_pressure_diastolic"]
                else None,
                respiratory_rate=int(request.form["respiratory_rate"])
                if request.form["respiratory_rate"]
                else None,
                oxygen_saturation=int(request.form["oxygen_saturation"])
                if request.form["oxygen_saturation"]
                else None,
                blood_glucose=float(request.form["blood_glucose"])
                if request.form["blood_glucose"]
                else None,
                weight=float(request.form["weight"])
                if request.form["weight"]
                else None,
                height=float(request.form["height"])
                if request.form["height"]
                else None,
            )
            db.session.add(vitals_data)
            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
            if waiting_entry:
                waiting_entry.seen = QueueStatus.VITALS_DONE
            db.session.commit()
            ScheduleEngine().mark_triage_complete(patient_id)
            logger.info(
                f"Nurse {current_user.id} recorded vitals for patient {patient_id}"
            )
            db.session.add(
                Log(
                    level="INFO",
                    message=f"Nurse {current_user.username} (ID: {current_user.id}) recorded vitals for patient {patient_id}",
                    user_id=current_user.id,
                    source="nursing",
                )
            )
            db.session.commit()
            flash("Vitals recorded successfully.", "success")
            return redirect(url_for("nursing.vitals", patient_id=patient_id))
        except ValueError as e:
            db.session.rollback()
            flash("Something went wrong. Please try again.", "error")
            logger.error(f"ValueError in nursing.vitals: {e}", exc_info=True)
            db.session.add(
                Log(
                    level="ERROR",
                    message=f"Invalid input error recording vitals: {str(e)}",
                    user_id=current_user.id,
                    source="nursing",
                )
            )
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            flash("Something went wrong. Please try again.", "error")
            logger.error(f"Error in nursing.vitals: {e}", exc_info=True)
            db.session.add(
                Log(
                    level="ERROR",
                    message=f"Error recording vitals: {str(e)}",
                    user_id=current_user.id,
                    source="nursing",
                )
            )
            db.session.commit()

    try:
        latest_vitals = (
            Vitals.query.filter_by(patient_id=patient_id)
            .order_by(Vitals.timestamp.desc())
            .first()
        )
        logger.info(f"Nurse {current_user.id} viewed vitals for patient {patient_id}")
        db.session.add(
            Log(
                level="INFO",
                message=f"Nurse {current_user.username} (ID: {current_user.id}) viewed vitals for patient {patient_id}",
                user_id=current_user.id,
                source="nursing",
            )
        )
        db.session.commit()
        return render_template(
            "nursing/vitals.html", patient_id=patient_id, vitals=latest_vitals
        )
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error(f"Error in nursing.vitals: {e}", exc_info=True)
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error loading vitals: {str(e)}",
                user_id=current_user.id,
                source="nursing",
            )
        )
        db.session.commit()
        return redirect(url_for("home"))


# Alert/Action line logic
def check_alert_action(dilation, time_hours):
    alert_dilation = 4 + time_hours  # Alert line: starts at 4 cm, increases 1 cm/hour
    action_dilation = (
        4 + (time_hours - 4) if time_hours >= 4 else 0
    )  # Action line: 4 hours to the right
    status = "Normal"
    if dilation < alert_dilation:
        status = "Crossed Alert Line"
    if time_hours >= 4 and dilation < action_dilation:
        status = "Crossed Action Line"
    return status


# Route to display the Partogram form
@bp.route("/partogram", methods=["GET"])
@login_required
@roles_required("nursing", "admin")
def record_partogram():
    """Displays the partogram form for recording patient data."""
    try:
        return render_template("nursing/partogram.html")
    except Exception as e:
        flash(f"Error loading partogram form: {str(e)}", "error")
        print(f"Debug: Error in nursing.record_partogram: {str(e)}")
        return redirect(url_for("nursing.index"))


# Route to submit the Partogram form
@bp.route("/submit_partogram", methods=["POST"])
@login_required
@roles_required("nursing", "admin")
def submit_partogram():
    """Handles submission of the partogram form and saves to the database."""

    errors = []

    try:
        # Extract form data
        patient_id = request.form.get("patient_id")
        fetal_heart_rate = request.form.get("fetal_heart_rate")
        amniotic_fluid = request.form.get("amniotic_fluid")
        moulding = request.form.get("moulding")
        cervical_dilation = request.form.get("cervical_dilation")
        time_hours = request.form.get("time_hours")
        contractions = request.form.get("contractions")
        oxytocin = request.form.get("oxytocin")
        drugs = request.form.get("drugs")
        pulse = request.form.get("pulse")
        bp = request.form.get("bp")
        temperature = request.form.get("temperature")
        urine_protein = request.form.get("urine_protein")
        urine_volume = request.form.get("urine_volume")
        urine_acetone = request.form.get("urine_acetone")
        timestamp = datetime.now(timezone.utc)

        # Step 1: Validate required fields
        required_fields = {
            "Patient ID": patient_id,
            "Fetal Heart Rate": fetal_heart_rate,
            "Amniotic Fluid Status": amniotic_fluid,
            "Moulding": moulding,
            "Cervical Dilation": cervical_dilation,
            "Time (hours)": time_hours,
            "Contractions": contractions,
            "Pulse": pulse,
            "Blood Pressure": bp,
            "Temperature": temperature,
            "Urine Protein": urine_protein,
            "Urine Volume": urine_volume,
            "Urine Acetone": urine_acetone,
        }
        for field_name, field_value in required_fields.items():
            if not field_value:
                errors.append(f"{field_name} is required.")

        # Step 2: Validate data types and convert
        if not errors:
            try:
                fetal_heart_rate = int(fetal_heart_rate)
                cervical_dilation = float(cervical_dilation)
                time_hours = float(time_hours)
                contractions = int(contractions)
                oxytocin = float(oxytocin) if oxytocin else 0.0
                pulse = int(pulse)
                if "/" not in bp:
                    errors.append(
                        "Blood Pressure must be in the format 'systolic/diastolic' (e.g., 120/80)."
                    )
                else:
                    bp_systolic, bp_diastolic = map(int, bp.split("/"))
                temperature = float(temperature)
                urine_volume = int(urine_volume)
            except ValueError as ve:
                errors.append(
                    f"Invalid numeric input: {str(ve)}. Please ensure all numeric fields contain valid numbers."
                )

        # Step 3: Validate ranges
        if not errors:
            if not (60 <= fetal_heart_rate <= 200):
                errors.append(
                    "Fetal Heart Rate must be between 60 and 200 beats per minute."
                )
            if not (0 <= cervical_dilation <= 10):
                errors.append("Cervical Dilation must be between 0 and 10 cm.")
            if not (0 <= time_hours):
                errors.append("Time since active labor start cannot be negative.")
            if not (0 <= contractions <= 5):
                errors.append("Contractions per 10 minutes must be between 0 and 5.")
            if oxytocin and oxytocin < 0:
                errors.append("Oxytocin units cannot be negative.")
            if not (40 <= pulse <= 200):
                errors.append("Pulse must be between 40 and 200 beats per minute.")
            if not (50 <= bp_systolic <= 200):
                errors.append(
                    "Systolic Blood Pressure must be between 50 and 200 mmHg."
                )
            if not (30 <= bp_diastolic <= 150):
                errors.append(
                    "Diastolic Blood Pressure must be between 30 and 150 mmHg."
                )
            if bp_systolic <= bp_diastolic:
                errors.append(
                    "Systolic Blood Pressure must be greater than Diastolic Blood Pressure."
                )
            if not (35 <= temperature <= 42):
                errors.append("Temperature must be between 35 and 42 °C.")
            if not (0 <= urine_volume):
                errors.append("Urine Volume cannot be negative.")

        # Step 4: Validate categorical fields
        if not errors:
            valid_amniotic_fluid = ["clear", "meconium", "blood", "absent"]
            if amniotic_fluid not in valid_amniotic_fluid:
                errors.append(
                    f"Amniotic Fluid Status must be one of: {', '.join(valid_amniotic_fluid)}."
                )

            valid_moulding = ["none", "1+", "2+", "3+"]
            if moulding not in valid_moulding:
                errors.append(f"Moulding must be one of: {', '.join(valid_moulding)}.")

            valid_urine_levels = ["negative", "1+", "2+", "3+"]
            if urine_protein not in valid_urine_levels:
                errors.append(
                    f"Urine Protein must be one of: {', '.join(valid_urine_levels)}."
                )
            if urine_acetone not in valid_urine_levels:
                errors.append(
                    f"Urine Acetone must be one of: {', '.join(valid_urine_levels)}."
                )

        # Step 5: Validate string lengths (optional fields)
        if not errors and drugs and len(drugs) > 100:
            errors.append("Drugs field cannot exceed 100 characters.")

        # Step 6: Check for increasing time_hours for the same patient
        if not errors:
            previous_entry = (
                Partogram.query.filter_by(patient_id=patient_id)
                .order_by(Partogram.time_hours.desc())
                .first()
            )
            if previous_entry and time_hours <= previous_entry.time_hours:
                errors.append(
                    "Time since active labor start must be greater than the previous entry for this patient."
                )

        # If there are errors, render the error page
        if errors:
            return render_template("nursing/error.html", errors=errors)

        # Step 7: Check alert/action lines
        labour_status = check_alert_action(cervical_dilation, time_hours)

        # Step 8: Create and save the Partogram entry
        new_partogram = Partogram(
            patient_id=patient_id,
            fetal_heart_rate=fetal_heart_rate,
            amniotic_fluid=amniotic_fluid,
            moulding=moulding,
            cervical_dilation=cervical_dilation,
            time_hours=time_hours,
            contractions=contractions,
            oxytocin=oxytocin,
            drugs=drugs,
            pulse=pulse,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            temperature=temperature,
            urine_protein=urine_protein,
            urine_volume=urine_volume,
            urine_acetone=urine_acetone,
            labour_status=labour_status,
            timestamp=timestamp,
            recorded_by=current_user.id,
        )

        db.session.add(new_partogram)
        db.session.commit()

        flash("Partogram recorded successfully!", "success")
        return redirect(url_for("nursing.view_partogram", patient_id=patient_id))

    except ValueError as ve:
        db.session.rollback()
        return render_template(
            "nursing/error.html", errors=[f"Invalid input: {str(ve)}"]
        )
    except Exception as e:
        db.session.rollback()
        return render_template(
            "nursing/error.html", errors=[f"Error saving partogram: {str(e)}"]
        )


@bp.route("/view_partogram/<patient_id>", methods=["GET"])
@login_required
@roles_required("nursing", "admin")
def view_partogram(patient_id):
    """Displays partogram records for a specific patient, including graphs."""
    try:
        records = (
            Partogram.query.filter_by(patient_id=patient_id)
            .order_by(Partogram.time_hours.asc())
            .all()
        )

        if not records:
            flash(f"No partogram records found for patient {patient_id}.", "info")
            return redirect(url_for("nursing.index"))

        # Convert ORM objects to dicts for template rendering
        entries = [
            {
                "time_hours": r.time_hours,
                "cervical_dilation": r.cervical_dilation,
                "fetal_heart_rate": r.fetal_heart_rate,
                "contractions": r.contractions,
                "pulse": r.pulse,
                "bp_systolic": r.bp_systolic,
                "bp_diastolic": r.bp_diastolic,
                "temperature": r.temperature,
                "amniotic_fluid": r.amniotic_fluid,
                "moulding": r.moulding,
                "labour_status": r.labour_status,
                "urine_protein": r.urine_protein,
                "urine_volume": r.urine_volume,
                "urine_acetone": r.urine_acetone,
                "timestamp": r.timestamp,
                "recorded_by": r.recorded_by,
            }
            for r in records
        ]

        # Build user_name_map via ORM — no raw SQL, no SQL injection risk
        recorded_by_ids = list(
            {entry["recorded_by"] for entry in entries if entry["recorded_by"]}
        )
        if recorded_by_ids:
            users = User.query.filter(User.id.in_(recorded_by_ids)).all()
            user_name_map = {u.id: u.username for u in users}
        else:
            user_name_map = {}

        logger.info(
            "Found %d partogram records for patient %s", len(entries), patient_id
        )

        return render_template(
            "nursing/view_partogram.html",
            partogram_data=entries,
            patient_id=patient_id,
            user_name_map=user_name_map,
        )

    except Exception as e:
        flash(f"Error fetching partogram records: {str(e)}", "error")
        logger.error("Error in nursing.view_partogram: %s", e, exc_info=True)
        return redirect(url_for("nursing.index"))


# Route to view all partograms (paginated)
@bp.route("/view_partograms")
@login_required
@roles_required("nursing", "admin")
def view_partograms():
    """Displays all partogram records, paginated, with optional patient_id filter."""
    patient_id_filter = request.args.get("patient_id", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = 10

    try:
        # Build a subquery: most-recent timestamp per patient
        from sqlalchemy import func

        latest_ts_sq = (
            db.session.query(
                Partogram.patient_id,
                func.max(Partogram.timestamp).label("max_ts"),
            )
            .group_by(Partogram.patient_id)
            .subquery()
        )

        query = Partogram.query.join(
            latest_ts_sq,
            (Partogram.patient_id == latest_ts_sq.c.patient_id)
            & (Partogram.timestamp == latest_ts_sq.c.max_ts),
        ).order_by(Partogram.timestamp.desc())

        if patient_id_filter:
            query = query.filter(Partogram.patient_id == patient_id_filter)

        partograms = query.paginate(page=page, per_page=per_page, error_out=False)

        # Build user_name_map for all recorded_by ids on this page
        recorded_by_ids = list(
            {p.recorded_by for p in partograms.items if p.recorded_by}
        )
        if recorded_by_ids:
            users = User.query.filter(User.id.in_(recorded_by_ids)).all()
            user_name_map = {u.id: u.username for u in users}
        else:
            user_name_map = {}

        return render_template(
            "nursing/view_partograms.html",
            partograms=partograms,
            user_name_map=user_name_map,
        )

    except Exception as e:
        logger.error("Error in nursing.view_partograms: %s", e, exc_info=True)
        return render_template(
            "nursing/error.html",
            errors=[f"Error fetching partogram records: {str(e)}"],
        )


@bp.route("/vital_signs", methods=["GET", "POST"])
@login_required
@roles_required("nursing", "admin", "medicine")
def vital_signs():
    if request.method == "POST":
        try:
            patient_id = request.form.get("patient_id")
            if not patient_id:
                flash("Patient ID is required.", "error")
                return redirect(url_for("nursing.vital_signs"))

            # Capture all fields from the Vitals model
            heart_rate = request.form.get("heart_rate")
            respiratory_rate = request.form.get("respiratory_rate")
            oxygen_saturation = request.form.get("oxygen_saturation")
            temperature = request.form.get("temperature")
            pulse = request.form.get("pulse")
            blood_pressure_systolic = request.form.get("blood_pressure_systolic")
            blood_pressure_diastolic = request.form.get("blood_pressure_diastolic")
            blood_glucose = request.form.get("blood_glucose")
            weight = request.form.get("weight")
            height = request.form.get("height")

            new_vital_sign = Vitals(
                patient_id=patient_id,
                heart_rate=int(heart_rate) if heart_rate else None,
                respiratory_rate=int(respiratory_rate) if respiratory_rate else None,
                oxygen_saturation=int(oxygen_saturation) if oxygen_saturation else None,
                temperature=float(temperature) if temperature else None,
                pulse=int(pulse) if pulse else None,
                blood_pressure_systolic=int(blood_pressure_systolic)
                if blood_pressure_systolic
                else None,
                blood_pressure_diastolic=int(blood_pressure_diastolic)
                if blood_pressure_diastolic
                else None,
                blood_glucose=float(blood_glucose) if blood_glucose else None,
                weight=float(weight) if weight else None,
                height=float(height) if height else None,
                timestamp=datetime.now(timezone.utc),
                recorded_by=current_user.id,
            )
            db.session.add(new_vital_sign)
            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
            if waiting_entry:
                waiting_entry.seen = QueueStatus.VITALS_DONE
            db.session.commit()
            ScheduleEngine().mark_triage_complete(patient_id)
            flash("Vital signs recorded.", "success")
            return redirect(url_for("nursing.vital_signs"))
        except ValueError:
            flash(
                "Invalid input: Please ensure numeric fields contain valid numbers.",
                "error",
            )
            return redirect(url_for("nursing.vital_signs"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error recording vital signs: {str(e)}", "error")
            return redirect(url_for("nursing.vital_signs"))

    return render_template("nursing/vital_signs.html")


@bp.route("/vitals_chart/<patient_id>")
@login_required
@roles_required("nursing", "admin", "medicine")
def vitals_chart(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    vitals_history = (
        Vitals.query.filter_by(patient_id=patient_id)
        .order_by(Vitals.timestamp.asc())
        .all()
    )

    timestamps = [v.timestamp.strftime("%m-%d %H:%M") for v in vitals_history]
    temps = [v.temperature or 0 for v in vitals_history]
    pulse = [v.pulse or 0 for v in vitals_history]
    bp_sys = [v.blood_pressure_systolic or 0 for v in vitals_history]
    bp_dia = [v.blood_pressure_diastolic or 0 for v in vitals_history]
    spo2 = [v.oxygen_saturation or 0 for v in vitals_history]

    return render_template(
        "nursing/vitals_chart.html",
        patient=patient,
        vitals_history=vitals_history,
        timestamps=timestamps,
        temps=temps,
        pulse=pulse,
        bp_sys=bp_sys,
        bp_dia=bp_dia,
        spo2=spo2,
    )


# ─────────────────────────────────────────────
# FLUID BALANCE CHART
# ─────────────────────────────────────────────


@bp.route("/fluid_balance/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required("nursing", "admin", "medicine")
def fluid_balance(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    return render_template("nursing/fluid_balance.html", patient=patient)
