from datetime import date, datetime, timedelta

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import extract, func
from sqlalchemy.orm import joinedload

from departments.api.audit import log_audit_event
from departments.models.laboratory import LabResult
from departments.models.medicine import (
    AdmittedPatient,
    PrescribedMedicine,
    RequestedImage,
    RequestedLab,
    SOAPNote,
)
from departments.models.nursing import NursingNote, Vitals
from departments.models.records import (
    Clinic,
    ClinicBooking,
    Patient,
    PatientAllergy,
    PatientProblem,
    PatientWaitingList,
)

# ─────────────────────────────────────────────
# INDEX — Patient List
# ─────────────────────────────────────────────
from departments.rbac import roles_required
from departments.records.merge import find_duplicate_candidates, merge_patient_records
from extensions import db

from . import bp


@bp.route("/index")
@login_required
@roles_required("records", "admin")
def index():
    patients = Patient.query.order_by(Patient.date_registered.desc()).all()
    clinics = Clinic.query.all()
    today = date.today()
    today_count = Patient.query.filter(
        func.date(Patient.date_registered) == today
    ).count()
    return render_template(
        "records/patients_list.html",
        patients=patients,
        clinics=clinics,
        today_count=today_count,
        total_count=len(patients),
    )


# ─────────────────────────────────────────────
# SEARCH (AJAX)
# ─────────────────────────────────────────────
@bp.route("/search_clinics")
@login_required
def search_clinics():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    clinics = Clinic.query.filter(Clinic.name.ilike(f"%{query}%")).limit(20).all()
    results = [clinic.to_select2() for clinic in clinics]
    return jsonify(results)


@bp.route("/search_patients", methods=["GET"])
@login_required
@roles_required("records", "admin")
def search_patients():
    search_term = request.args.get("term", "").strip()
    if search_term:
        patients = Patient.query.filter(
            Patient.name.ilike(f"%{search_term}%")
            | Patient.patient_id.ilike(f"%{search_term}%")
        ).all()
    else:
        patients = []
    results = [
        {"id": p.patient_id, "text": f"{p.name} ({p.patient_id})"} for p in patients
    ]
    return jsonify({"results": results})


# ─────────────────────────────────────────────
# NEW PATIENT
# ─────────────────────────────────────────────
@bp.route("/new_patient", methods=["GET", "POST"])
@login_required
@roles_required("records", "admin")
def new_patient():
    if request.method == "POST":
        name = request.form["name"]
        place_of_residence = request.form["place_of_residence"]
        sex = request.form["sex"]
        date_of_birth = request.form["date_of_birth"]
        marital_status = request.form["marital_status"]
        blood_group = request.form["blood_group"]
        contact = request.form["contact"]
        next_of_kin = request.form["next_of_kin"]
        relationship_with_next_of_kin = request.form["relationship_with_next_of_kin"]
        next_of_kin_contact = request.form["next_of_kin_contact"]
        national_id = request.form["national_id"]
        insurance_provider = request.form.get("insurance_provider", None)
        insurance_policy_number = request.form.get("insurance_policy_number", None)
        occupation = request.form.get("occupation", None)
        employer_name = request.form.get("employer_name", None)
        emergency_contact = request.form["emergency_contact"]

        if not all(
            [
                name,
                place_of_residence,
                sex,
                date_of_birth,
                marital_status,
                blood_group,
                contact,
                next_of_kin,
                relationship_with_next_of_kin,
                next_of_kin_contact,
                national_id,
                emergency_contact,
            ]
        ):
            flash("All required fields must be filled!", "danger")
            return redirect(url_for("records.new_patient"))

        try:
            date_of_birth = datetime.strptime(date_of_birth, "%Y-%m-%d").date()
        except ValueError:
            flash("Invalid date format for Date of Birth! Use YYYY-MM-DD.", "danger")
            return redirect(url_for("records.new_patient"))

        new_patient_id = Patient.generate_patient_id(Patient())

        new_p = Patient(
            patient_id=new_patient_id,
            name=name,
            place_of_residence=place_of_residence,
            sex=sex,
            date_of_birth=date_of_birth,
            marital_status=marital_status,
            blood_group=blood_group,
            contact=contact,
            next_of_kin=next_of_kin,
            relationship_with_next_of_kin=relationship_with_next_of_kin,
            next_of_kin_contact=next_of_kin_contact,
            national_id=national_id,
            insurance_provider=insurance_provider,
            insurance_policy_number=insurance_policy_number,
            occupation=occupation,
            employer_name=employer_name,
            emergency_contact=emergency_contact,
        )
        db.session.add(new_p)
        db.session.commit()

        waiting_entry = PatientWaitingList(patient_id=new_p.patient_id, seen=4)
        db.session.add(waiting_entry)
        db.session.commit()

        flash(
            f"Patient {new_p.name} registered successfully with ID: {new_p.patient_id}!",
            "success",
        )
        log_audit_event(
            "PATIENT_CREATE",
            resource_type="Patient",
            resource_id=new_p.patient_id,
            details={"name": new_p.name},
        )
        return redirect(url_for("records.patient_profile", patient_id=new_p.patient_id))

    return render_template("records/new_patient.html")


# ─────────────────────────────────────────────
# PATIENT PROFILE
# ─────────────────────────────────────────────
@bp.route("/patient/<patient_id>")
@login_required
@roles_required("records", "admin")
def patient_profile(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    log_audit_event("PATIENT_VIEW", resource_type="Patient", resource_id=patient_id)
    return render_template("records/patient_profile.html", patient=patient)


# ─────────────────────────────────────────────
# EDIT PATIENT
# ─────────────────────────────────────────────
@bp.route("/patient/<patient_id>/edit", methods=["GET", "POST"])
@login_required
@roles_required("records", "admin")
def edit_patient(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()

    if request.method == "POST":
        patient.name = request.form["name"]
        patient.place_of_residence = request.form["place_of_residence"]
        patient.sex = request.form["sex"]
        try:
            patient.date_of_birth = datetime.strptime(
                request.form["date_of_birth"], "%Y-%m-%d"
            ).date()
        except ValueError:
            flash("Invalid date format!", "danger")
            return redirect(url_for("records.edit_patient", patient_id=patient_id))
        patient.marital_status = request.form["marital_status"]
        patient.blood_group = request.form["blood_group"]
        patient.contact = request.form["contact"]
        patient.next_of_kin = request.form["next_of_kin"]
        patient.relationship_with_next_of_kin = request.form[
            "relationship_with_next_of_kin"
        ]
        patient.next_of_kin_contact = request.form["next_of_kin_contact"]
        patient.national_id = request.form["national_id"]
        patient.insurance_provider = request.form.get("insurance_provider")
        patient.insurance_policy_number = request.form.get("insurance_policy_number")
        patient.occupation = request.form.get("occupation")
        patient.employer_name = request.form.get("employer_name")
        patient.emergency_contact = request.form["emergency_contact"]
        db.session.commit()
        log_audit_event(
            "PATIENT_UPDATE", resource_type="Patient", resource_id=patient_id
        )
        flash("Patient details updated successfully!", "success")
        return redirect(url_for("records.patient_profile", patient_id=patient_id))

    return render_template("records/edit_patient.html", patient=patient)


# ─────────────────────────────────────────────
# PATIENT MEDICAL HISTORY
# ─────────────────────────────────────────────
@bp.route("/patient/<patient_id>/history")
@login_required
@roles_required("records", "admin")
def patient_history(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()

    soap_notes = (
        SOAPNote.query.filter_by(patient_id=patient_id)
        .order_by(SOAPNote.created_at.desc())
        .all()
    )

    prescribed = (
        PrescribedMedicine.query.filter_by(patient_id=patient_id)
        .order_by(PrescribedMedicine.id.desc())
        .all()
    )

    requested_labs = (
        RequestedLab.query.filter_by(patient_id=patient_id)
        .order_by(RequestedLab.date_requested.desc())
        .all()
    )

    lab_results = (
        LabResult.query.filter_by(patient_id=patient_id)
        .order_by(LabResult.test_date.desc())
        .all()
    )

    requested_images = (
        RequestedImage.query.filter_by(patient_id=patient_id)
        .order_by(RequestedImage.date_requested.desc())
        .all()
    )

    vitals = (
        Vitals.query.filter_by(patient_id=patient_id)
        .order_by(Vitals.timestamp.desc())
        .all()
    )

    nursing_notes = (
        NursingNote.query.filter_by(patient_id=patient_id)
        .order_by(NursingNote.timestamp.desc())
        .all()
    )

    admissions = (
        AdmittedPatient.query.filter_by(patient_id=patient_id)
        .order_by(AdmittedPatient.admitted_on.desc())
        .all()
    )

    clinic_bookings = (
        ClinicBooking.query.filter_by(patient_id=patient_id)
        .order_by(ClinicBooking.clinic_date.desc())
        .all()
    )

    allergies = (
        PatientAllergy.query.filter_by(patient_id=patient_id)
        .order_by(PatientAllergy.date_recorded.desc())
        .all()
    )

    problems = (
        PatientProblem.query.filter_by(patient_id=patient_id)
        .order_by(PatientProblem.created_at.desc())
        .all()
    )

    return render_template(
        "records/patient_history.html",
        patient=patient,
        soap_notes=soap_notes,
        prescribed=prescribed,
        requested_labs=requested_labs,
        lab_results=lab_results,
        requested_images=requested_images,
        vitals=vitals,
        nursing_notes=nursing_notes,
        admissions=admissions,
        clinic_bookings=clinic_bookings,
        allergies=allergies,
        problems=problems,
    )


# ─────────────────────────────────────────────
# PATIENT ALLERGY REGISTRY & PROBLEM LIST APIs
# ─────────────────────────────────────────────
@bp.route("/patient/<patient_id>/allergies", methods=["GET", "POST"])
@login_required
def manage_patient_allergies(patient_id):
    Patient.query.filter_by(patient_id=patient_id).first_or_404()
    if request.method == "POST":
        data = request.get_json() or request.form
        allergen = data.get("allergen")
        if not allergen:
            return jsonify({"error": "Allergen name required"}), 400
        category = data.get("category", "DRUG")
        reaction = data.get("reaction", "")
        severity = data.get("severity", "MODERATE")

        allergy = PatientAllergy(
            patient_id=patient_id,
            allergen=allergen,
            category=category,
            reaction=reaction,
            severity=severity,
            recorded_by=getattr(current_user, "id", None),
        )
        db.session.add(allergy)
        db.session.commit()
        log_audit_event(
            "PATIENT_ALLERGY_ADD",
            resource_type="PatientAllergy",
            resource_id=str(allergy.id),
            details={"allergen": allergen, "patient_id": patient_id},
        )
        return jsonify(
            {
                "success": True,
                "allergy_id": allergy.id,
                "message": "Allergy record added successfully.",
            }
        ), 201

    allergies = (
        PatientAllergy.query.filter_by(patient_id=patient_id)
        .order_by(PatientAllergy.date_recorded.desc())
        .all()
    )
    return jsonify(
        [
            {
                "id": a.id,
                "allergen": a.allergen,
                "category": a.category,
                "reaction": a.reaction,
                "severity": a.severity,
                "date_recorded": a.date_recorded.isoformat()
                if a.date_recorded
                else None,
            }
            for a in allergies
        ]
    ), 200


@bp.route("/patient/<patient_id>/problems", methods=["GET", "POST"])
@login_required
def manage_patient_problems(patient_id):
    Patient.query.filter_by(patient_id=patient_id).first_or_404()
    if request.method == "POST":
        data = request.get_json() or request.form
        description = data.get("description")
        if not description:
            return jsonify({"error": "Problem description required"}), 400
        icd10_code = data.get("icd10_code")
        status = data.get("status", "ACTIVE")

        problem = PatientProblem(
            patient_id=patient_id,
            icd10_code=icd10_code,
            description=description,
            status=status,
            created_by=getattr(current_user, "id", None),
        )
        db.session.add(problem)
        db.session.commit()
        log_audit_event(
            "PATIENT_PROBLEM_ADD",
            resource_type="PatientProblem",
            resource_id=str(problem.id),
            details={
                "description": description,
                "status": status,
                "patient_id": patient_id,
            },
        )
        return jsonify(
            {
                "success": True,
                "problem_id": problem.id,
                "message": "Active problem recorded successfully.",
            }
        ), 201

    problems = (
        PatientProblem.query.filter_by(patient_id=patient_id)
        .order_by(PatientProblem.created_at.desc())
        .all()
    )
    return jsonify(
        [
            {
                "id": p.id,
                "icd10_code": p.icd10_code,
                "description": p.description,
                "status": p.status,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in problems
        ]
    ), 200


@bp.route(
    "/patient/<patient_id>/problems/<int:problem_id>/status", methods=["POST", "PUT"]
)
@login_required
def update_patient_problem_status(patient_id, problem_id):
    problem = PatientProblem.query.filter_by(
        id=problem_id, patient_id=patient_id
    ).first_or_404()
    data = request.get_json() or request.form
    new_status = data.get("status", "RESOLVED").upper()
    if new_status not in ("ACTIVE", "RESOLVED", "CHRONIC"):
        return jsonify(
            {"error": "Invalid status. Must be ACTIVE, RESOLVED, or CHRONIC"}
        ), 400

    problem.status = new_status
    if new_status == "RESOLVED":
        problem.resolved_date = date.today()
    db.session.commit()
    log_audit_event(
        "PATIENT_PROBLEM_UPDATE",
        resource_type="PatientProblem",
        resource_id=str(problem.id),
        details={"new_status": new_status, "patient_id": patient_id},
    )
    return jsonify(
        {"success": True, "problem_id": problem.id, "status": problem.status}
    ), 200


# ─────────────────────────────────────────────
# CLINICS
# ─────────────────────────────────────────────
@bp.route("/clinics")
@login_required
@roles_required("records", "admin")
def clinics():
    clinics = Clinic.query.all()
    return render_template("records/clinics_list.html", clinics=clinics)


@bp.route("/clinics/add", methods=["GET", "POST"])
@login_required
@roles_required("records", "admin")
def add_clinic():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        fee = request.form.get("fee", "").strip()
        if not name or not fee:
            flash("Clinic name and fee are required.", "danger")
            return redirect(url_for("records.add_clinic"))
        try:
            fee = float(fee)
        except ValueError:
            flash("Fee must be a valid number.", "danger")
            return redirect(url_for("records.add_clinic"))
        clinic = Clinic(name=name, fee=fee)
        db.session.add(clinic)
        db.session.commit()
        flash(f'Clinic "{name}" added successfully!', "success")
        return redirect(url_for("records.clinics"))
    return render_template("records/add_clinic.html")


@bp.route("/clinics/<int:clinic_id>/edit", methods=["GET", "POST"])
@login_required
@roles_required("records", "admin")
def edit_clinic(clinic_id):
    clinic = Clinic.query.get_or_404(clinic_id)
    if request.method == "POST":
        clinic.name = request.form.get("name", clinic.name).strip()
        try:
            clinic.fee = float(request.form.get("fee", clinic.fee))
        except ValueError:
            flash("Fee must be a valid number.", "danger")
            return redirect(url_for("records.edit_clinic", clinic_id=clinic_id))
        db.session.commit()
        flash("Clinic updated successfully!", "success")
        return redirect(url_for("records.clinics"))
    return render_template("records/add_clinic.html", clinic=clinic)


# ─────────────────────────────────────────────
# BOOKINGS
# ─────────────────────────────────────────────
@bp.route("/bookings")
@login_required
@roles_required("records", "admin")
def bookings():
    bookings = (
        db.session.query(ClinicBooking)
        .options(joinedload(ClinicBooking.patient), joinedload(ClinicBooking.clinic))
        .order_by(ClinicBooking.clinic_date.desc())
        .all()
    )
    return render_template("records/bookings_list.html", bookings=bookings)


@bp.route("/book_clinic", methods=["POST"])
@login_required
@roles_required("records", "admin")
def book_clinic():
    patient_id = request.form.get("patient_id")
    clinic_id = request.form.get("clinic_id")
    clinic_date = request.form.get("clinic_date")

    if not patient_id or not clinic_id or not clinic_date:
        return jsonify({"status": "error", "message": "All fields are required!"}), 400

    try:
        clinic_date = datetime.strptime(clinic_date, "%Y-%m-%d").date()
    except ValueError:
        return jsonify(
            {"status": "error", "message": "Invalid date format! Use YYYY-MM-DD."}
        ), 400

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify(
            {
                "status": "error",
                "message": f"Patient with ID {patient_id} does not exist!",
            }
        ), 400

    clinic = Clinic.query.get(clinic_id)
    if not clinic:
        return jsonify(
            {
                "status": "error",
                "message": f"Clinic with ID {clinic_id} does not exist!",
            }
        ), 400

    existing_booking = ClinicBooking.query.filter_by(
        patient_id=patient_id, clinic_id=clinic_id, clinic_date=clinic_date
    ).first()
    if existing_booking:
        return jsonify(
            {"status": "error", "message": "This booking already exists!"}
        ), 400

    new_booking = ClinicBooking(
        patient_id=patient_id, clinic_id=clinic_id, clinic_date=clinic_date
    )
    db.session.add(new_booking)

    waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
    if not waiting_entry:
        waiting_entry = PatientWaitingList(patient_id=patient_id, seen=4)
        db.session.add(waiting_entry)
    else:
        waiting_entry.seen = 4

    db.session.commit()
    return jsonify(
        {
            "status": "success",
            "message": f"Clinic booked for {patient.name} at {clinic.name} on {clinic_date.strftime('%Y-%m-%d')}!",
        }
    ), 200


# ─────────────────────────────────────────────
# WAITING LIST
# ─────────────────────────────────────────────
@bp.route("/waiting_list")
@login_required
@roles_required("records", "admin")
def waiting_list():
    waiting_list = (
        db.session.query(PatientWaitingList, Patient)
        .join(Patient, PatientWaitingList.patient_id == Patient.patient_id)
        .all()
    )
    return render_template("records/waiting_list.html", waiting_list=waiting_list)


# ─────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────
@bp.route("/reports/daily_opd")
@login_required
@roles_required("records", "admin")
def daily_opd_report():
    # Date range filter
    start_str = request.args.get("start")
    end_str = request.args.get("end")

    try:
        start_date = (
            datetime.strptime(start_str, "%Y-%m-%d").date()
            if start_str
            else date.today() - timedelta(days=29)
        )
        end_date = (
            datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else date.today()
        )
    except ValueError:
        start_date = date.today() - timedelta(days=29)
        end_date = date.today()

    # OPD = patients registered (waiting list created) per day
    # Use PatientWaitingList as proxy for OPD visits (new registrations)
    # Also count clinic bookings as OPD attendance
    daily_registrations = (
        db.session.query(
            func.date(Patient.date_registered).label("reg_date"),
            func.count(Patient.id).label("count"),
        )
        .filter(
            func.date(Patient.date_registered) >= start_date,
            func.date(Patient.date_registered) <= end_date,
        )
        .group_by(func.date(Patient.date_registered))
        .order_by(func.date(Patient.date_registered))
        .all()
    )

    daily_bookings = (
        db.session.query(
            ClinicBooking.clinic_date.label("visit_date"),
            func.count(ClinicBooking.id).label("count"),
        )
        .filter(
            ClinicBooking.clinic_date >= start_date,
            ClinicBooking.clinic_date <= end_date,
        )
        .group_by(ClinicBooking.clinic_date)
        .order_by(ClinicBooking.clinic_date)
        .all()
    )

    total_registrations = sum(r.count for r in daily_registrations)
    total_bookings = sum(b.count for b in daily_bookings)

    return render_template(
        "records/reports_daily_opd.html",
        daily_registrations=daily_registrations,
        daily_bookings=daily_bookings,
        total_registrations=total_registrations,
        total_bookings=total_bookings,
        start_date=start_date,
        end_date=end_date,
    )


@bp.route("/reports/clinic_attendance")
@login_required
@roles_required("records", "admin")
def clinic_attendance_report():
    start_str = request.args.get("start")
    end_str = request.args.get("end")
    try:
        start_date = (
            datetime.strptime(start_str, "%Y-%m-%d").date()
            if start_str
            else date.today().replace(day=1)
        )
        end_date = (
            datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else date.today()
        )
    except ValueError:
        start_date = date.today().replace(day=1)
        end_date = date.today()

    clinics = Clinic.query.all()
    attendance_data = []
    for clinic in clinics:
        total = ClinicBooking.query.filter(
            ClinicBooking.clinic_id == clinic.clinic_id,
            ClinicBooking.clinic_date >= start_date,
            ClinicBooking.clinic_date <= end_date,
        ).count()
        seen = ClinicBooking.query.filter(
            ClinicBooking.clinic_id == clinic.clinic_id,
            ClinicBooking.clinic_date >= start_date,
            ClinicBooking.clinic_date <= end_date,
            ClinicBooking.seen == 1,
        ).count()
        attendance_data.append(
            {
                "clinic": clinic,
                "total": total,
                "seen": seen,
                "not_seen": total - seen,
                "rate": round((seen / total * 100) if total > 0 else 0, 1),
            }
        )

    return render_template(
        "records/reports_clinic_attendance.html",
        attendance_data=attendance_data,
        start_date=start_date,
        end_date=end_date,
    )


@bp.route("/reports/registrations")
@login_required
@roles_required("records", "admin")
def registration_report():
    year_str = request.args.get("year")
    try:
        year = int(year_str) if year_str else date.today().year
    except ValueError:
        year = date.today().year

    monthly_data = (
        db.session.query(
            extract("month", Patient.date_registered).label("month"),
            func.count(Patient.id).label("count"),
        )
        .filter(extract("year", Patient.date_registered) == year)
        .group_by(extract("month", Patient.date_registered))
        .order_by(extract("month", Patient.date_registered))
        .all()
    )

    # Build full 12-month array
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_counts = [0] * 12
    for row in monthly_data:
        month_counts[int(row.month) - 1] = row.count

    total_year = sum(month_counts)
    sex_data = (
        db.session.query(Patient.sex, func.count(Patient.id).label("count"))
        .filter(extract("year", Patient.date_registered) == year)
        .group_by(Patient.sex)
        .all()
    )

    available_years = (
        db.session.query(extract("year", Patient.date_registered).label("yr"))
        .distinct()
        .order_by(extract("year", Patient.date_registered).desc())
        .all()
    )
    available_years = [int(r.yr) for r in available_years]

    return render_template(
        "records/reports_registrations.html",
        month_names=month_names,
        month_counts=month_counts,
        total_year=total_year,
        sex_data=sex_data,
        year=year,
        available_years=available_years,
    )


# ─────────────────────────────────────────────
# PATIENT MERGE — Duplicate Detection & Merge
# ─────────────────────────────────────────────


@bp.route("/patient/<patient_id>/duplicates")
@login_required
@roles_required("records", "admin")
def patient_duplicates(patient_id):
    """Show potential duplicate records for a given patient."""
    patient = Patient.query.filter_by(
        patient_id=patient_id, is_active=True
    ).first_or_404()
    candidates = find_duplicate_candidates(patient)
    return render_template(
        "records/patient_duplicates.html", patient=patient, candidates=candidates
    )


@bp.route("/patient/<patient_id>/merge", methods=["GET", "POST"])
@login_required
@roles_required("records", "admin")
def merge_patient(patient_id):
    """
    GET  – Show confirmation page before merging source into target.
    POST – Execute the merge: soft-delete source, attach audit log.

    Query param `target_id` required for both methods.
    """
    source = Patient.query.filter_by(
        patient_id=patient_id, is_active=True
    ).first_or_404()
    target_id = request.args.get("target_id") or request.form.get("target_id")
    if not target_id:
        flash("Target patient ID is required for a merge.", "danger")
        return redirect(url_for("records.patient_duplicates", patient_id=patient_id))

    target = Patient.query.filter_by(
        patient_id=target_id, is_active=True
    ).first_or_404()

    if request.method == "POST":
        notes = request.form.get("notes", "").strip()
        try:
            merge_patient_records(
                source_patient_id=patient_id,
                target_patient_id=target_id,
                user_id=current_user.id,
                notes=notes or None,
            )
            flash(
                f"Patient {source.name} ({patient_id}) merged into "
                f"{target.name} ({target_id}) successfully.",
                "success",
            )
            return redirect(url_for("records.patient_profile", patient_id=target_id))
        except ValueError as e:
            flash(str(e), "danger")
            return redirect(
                url_for("records.patient_duplicates", patient_id=patient_id)
            )

    return render_template("records/confirm_merge.html", source=source, target=target)


@bp.route("/api/patient/<patient_id>/soft-delete", methods=["POST"])
@login_required
@roles_required("admin")
def soft_delete_patient(patient_id):
    """Soft-delete a patient record (admin-only JSON endpoint)."""
    patient = Patient.query.filter_by(
        patient_id=patient_id, is_active=True
    ).first_or_404()
    patient.soft_delete()
    db.session.commit()
    return jsonify({"status": "ok", "message": f"Patient {patient_id} soft-deleted."})
