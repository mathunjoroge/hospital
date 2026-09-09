import os
import uuid
from datetime import datetime, timezone

from flask import flash, redirect, render_template, request, session, url_for
from flask_login import login_required
from psycopg2.extras import RealDictCursor

from departments.clinical_safety.engine import ClinicalSafetyEngine
from departments.consent.models import Consent
from departments.medicine.orders import fetch_drugs_data
from departments.models.medicine import (
    AdmittedPatient,
    Medicine,
    OncologyBooking,
    OncologyDrug,
    OncologyRegimen,
    OncoPrescription,
    PrescribedMedicine,
    PrescriptionDrugDetail,
    RegimenDrugAssociation,
)
from departments.models.records import Patient, PatientWaitingList
from departments.nlp.chatbot import UniversalClinicalSummarizer
from departments.nlp.logging_setup import get_logger
from departments.rbac import roles_required
from departments.shared.drugcentral import (
    get_drugcentral_connection as get_db_connection,
)
from extensions import db

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


@bp.route("/prescribe_drugs/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def prescribe_drugs(patient_id):
    """Handles drug prescription requests for admitted and waiting list patients."""

    try:
        patient_id = str(patient_id).strip()
        logger.debug(f"Checking patient_id={patient_id}")

        # Get department from GET or POST
        dept = request.args.get("dept") or request.form.get("dept")

        # Check if patient is admitted or in waiting list
        admitted_patient = AdmittedPatient.query.filter_by(
            patient_id=patient_id, discharged_on=None
        ).first()
        waiting_patient = PatientWaitingList.query.filter_by(
            patient_id=patient_id
        ).first()

        if not admitted_patient and not waiting_patient:
            flash(
                f"Patient with ID {patient_id} is neither admitted nor in the waiting list.",
                "error",
            )
            return redirect(url_for("medicine.index"))

        # Get patient record
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f"Patient with ID {patient_id} not found in the system!", "error")
            return redirect(url_for("medicine.index"))

        # Generate new prescription session if missing
        if "prescription_id" not in session:
            session["prescription_id"] = str(uuid.uuid4())
        prescription_id = session["prescription_id"]

        drugs = Medicine.query.all()

        if request.method == "POST":
            drugs_selected = request.form.getlist("drugs[]")
            dosage_form = request.form.get("dosage_form")
            strength = request.form.get("strength")
            frequency = request.form.get("frequency")
            custom_frequency = request.form.get("custom_frequency")
            num_days = request.form.get("num_days")

            # Validation
            if not all([drugs_selected, dosage_form, strength, frequency, num_days]):
                flash("All fields are required for prescribing drugs!", "error")
                return render_template(
                    "medicine/prescribe_drugs.html",
                    patient=patient,
                    drugs=drugs,
                    prescription_id=prescription_id,
                    prescribed_medicines=[],
                    dept=dept,
                )

            # Save prescriptions
            for drug_id in drugs_selected:
                prescribed = PrescribedMedicine(
                    patient_id=patient.patient_id,
                    medicine_id=drug_id,
                    dosage=dosage_form,
                    strength=strength.strip(),
                    frequency=custom_frequency.strip()
                    if frequency == "Other"
                    else frequency,
                    prescription_id=prescription_id,
                    num_days=int(num_days),
                )
                db.session.add(prescribed)

            try:
                db.session.commit()

            except Exception as e:
                db.session.rollback()
                logger.error(f"Database commit failed: {e}")
                flash("Something went wrong. Please try again.", "error")
                return redirect(
                    url_for(
                        "medicine.prescribe_drugs", patient_id=patient_id, dept=dept
                    )
                )

        # Consent check
        patient_int_id = None
        try:
            patient_int_id = int(patient_id)
        except (ValueError, TypeError):
            pass

        consent_rec = None
        if patient_int_id:
            consent_rec = Consent.query.filter_by(
                patient_id=patient_int_id, consent_type="TREATMENT"
            ).first()
        consent_status = (
            "ACTIVE"
            if (consent_rec and consent_rec.status == "ACTIVE" and consent_rec.revoked_at is None)
            else "MISSING"
        )

        # CDS safety check
        cds_alerts = []
        if patient_int_id:
            drug_ids_to_check = []
            if request.method == "POST":
                for d in request.form.getlist("drugs[]"):
                    try:
                        drug_ids_to_check.append(int(d))
                    except (ValueError, TypeError):
                        pass
            engine = ClinicalSafetyEngine()
            result = engine.check_prescription(
                patient_id=patient_int_id, drug_ids=drug_ids_to_check
            )
            cds_alerts = [a.to_dict() for a in result.alerts]

        # Fetch prescribed medicines
        prescribed_medicines = PrescribedMedicine.query.filter_by(
            prescription_id=prescription_id
        ).all()

        return render_template(
            "medicine/prescribe_drugs.html",
            patient=patient,
            drugs=drugs,
            prescribed_medicines=prescribed_medicines,
            prescription_id=prescription_id,
            dept=dept,
            consent_status=consent_status,
            cds_alerts=cds_alerts,
        )

    except Exception:
        db.session.rollback()
        logger.exception("Unexpected error in prescribe_drugs")
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("medicine.index"))


@bp.route("/edit_prescribed_medicine/<medicine_id>", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def edit_prescribed_medicine(medicine_id):
    """Handles editing a prescribed medicine."""
    try:
        # Get dept from either GET or POST
        dept = request.args.get("dept") or request.form.get("dept")
        prescription_id = request.form.get("prescription_id") or request.args.get(
            "prescription_id"
        )

        prescribed_medicine = PrescribedMedicine.query.get(medicine_id)
        if not prescribed_medicine:
            flash("Prescribed medicine not found.", "error")
            return redirect(url_for("medicine.index"))

        if request.method == "POST":
            dosage_form = request.form.get("dosage_form")
            custom_dosage = request.form.get("custom_dosage")
            strength = request.form.get("strength")
            frequency = request.form.get("frequency")
            custom_frequency = request.form.get("custom_frequency")
            num_days = request.form.get("num_days")

            if not all([dosage_form, strength, frequency, num_days]):
                flash("All fields are required!", "error")
                return redirect(
                    url_for(
                        "medicine.edit_prescribed_medicine",
                        medicine_id=medicine_id,
                        prescription_id=prescription_id,
                        dept=dept,
                    )
                )

            prescribed_medicine.dosage = (
                custom_dosage.strip() if dosage_form == "Other" else dosage_form
            )
            prescribed_medicine.strength = strength.strip()
            prescribed_medicine.frequency = (
                custom_frequency.strip() if frequency == "Other" else frequency
            )
            prescribed_medicine.num_days = int(num_days)
            db.session.commit()
            flash("Prescribed medicine updated successfully!", "success")
            return redirect(
                url_for(
                    "medicine.prescribe_drugs",
                    patient_id=prescribed_medicine.patient_id,
                    dept=dept,
                )
            )

        return render_template(
            "medicine/edit_prescribed_medicine.html",
            prescribed_medicine=prescribed_medicine,
            dept=dept,
        )

    except Exception:
        logger.exception("Error editing prescribed medicine")
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("medicine.index"))


@bp.route("/delete_prescribed_medicine/<medicine_id>", methods=["POST"])
@login_required
@roles_required("medicine", "admin")
def delete_prescribed_medicine(medicine_id):
    """Handles deleting a prescribed medicine."""
    # Get from GET or POST
    prescription_id = request.args.get("prescription_id") or request.form.get(
        "prescription_id"
    )
    dept = request.args.get("dept") or request.form.get("dept")

    try:
        prescribed_medicine = PrescribedMedicine.query.get_or_404(medicine_id)
        patient_id = prescribed_medicine.patient_id
        db.session.delete(prescribed_medicine)
        db.session.commit()
        flash("Prescribed medicine deleted successfully!", "success")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting prescribed medicine: {e}")
        flash("Something went wrong. Please try again.", "error")
        patient_id = prescribed_medicine.patient_id if prescribed_medicine else None

    return redirect(
        url_for(
            "medicine.prescribe_drugs",
            patient_id=patient_id,
            prescription_id=prescription_id,
            dept=dept,
        )
    )


@bp.route("/save_prescription/<prescription_id>/<patient_id>", methods=["POST"])
@login_required
@roles_required("medicine", "admin")
def save_prescription(prescription_id, patient_id):
    """Finalize the prescription and redirect appropriately based on dept."""
    try:
        # Get optional dept from query or form
        dept = request.args.get("dept") or request.form.get("dept")

        # Optional: mark prescription as finalized in DB here

        # ✅ Clear current prescription session
        session.pop("prescription_id", None)

        flash("Prescription finalized and saved successfully.", "success")

        # Redirect based on dept value
        if dept == "1":
            return redirect(url_for("medicine.ward_rounds"))
        else:
            return redirect(url_for("medicine.index"))

    except Exception as e:
        logger.error(f"Error finalizing prescription: {e}")
        flash("Something went wrong. Please try again.", "error")
        return redirect(
            url_for("medicine.prescribe_drugs", patient_id=patient_id, dept=dept)
        )


@bp.route("/get_edit_form", methods=["GET"])
@login_required
@roles_required("medicine", "admin")
def get_edit_form():
    """Serves the edit form for a prescribed medicine."""
    try:
        medicine_id = request.args.get("medicine_id")
        prescribed_medicine = PrescribedMedicine.query.get_or_404(medicine_id)
        return render_template(
            "medicine/edit_prescribed_medicine.html",
            prescribed_medicine=prescribed_medicine,
        )
    except Exception as e:
        print(f"Debug: Error in medicine.get_edit_form: {e}")
        return "Error loading edit form."


@bp.route("/drugs-ref/search", methods=["GET"])
def drugs_ref():
    """Drugs reference route with search functionality."""
    search_query = request.args.get("search", "").strip()
    drugs_data = fetch_drugs_data(search_query)

    return render_template(
        "medicine/drugs_ref.html", drugs_data=drugs_data, search_query=search_query
    )


@bp.route("/drugs-ref/details/<drug>", methods=["GET"])
def drug_details(drug: str):
    """Fetch and display detailed information about a specific active ingredient."""
    try:
        with get_db_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            # Normalize the drug name to uppercase for consistency
            normalized_drug = drug.upper()

            # Table name mapping for user-friendly display
            TABLE_NAME_MAPPING = {
                "faers": "Side Effects",
                "faers_male": "Side Effects (Male)",
                "faers_female": "Side Effects (Female)",
                "faers_ped": "Side Effects (Pediatric)",
                "faers_ger": "Side Effects (Geriatric)",
                "approval": "Regulatory Approvals",
                "ob_patent_view": "Patents",
                "ob_exclusivity_view": "Exclusivity Data",
                "active_ingredient": "Active Ingredients",
                "pharma_class": "Pharmacological Class",
                "act_table_full": "Drug-Target Interactions",
                "pka": "pKa Values",
                "pdb": "Protein Data Bank (PDB) Structures",
                "atc_ddd": "ATC Classification & Defined Daily Dose",
                "struct2obprod": "Marketed Drug Products",
                "struct2atc": "ATC Codes",
                "omop_relationship": "Clinical Data Relationships",
            }

            # Step 1: Fetch struct_id from the active_ingredient table
            cur.execute(
                """
                SELECT DISTINCT struct_id
                FROM active_ingredient
                WHERE UPPER(substance_name) = %s
            """,
                [normalized_drug],
            )
            result = cur.fetchone()

            if not result:
                return render_template(
                    "medicine/error.html", message=f"No details found for {drug}."
                )

            struct_id = result["struct_id"]

            # Step 2: Fetch data from all tables using struct_id
            query = """
                SELECT 'active_ingredient' AS table_name, struct_id::TEXT, substance_name::TEXT, quantity::TEXT, unit::TEXT, NULL::TEXT
                FROM active_ingredient WHERE struct_id = %s
                UNION ALL
                SELECT 'approval' AS table_name, struct_id::TEXT, approval::TEXT, applicant::TEXT, type::TEXT, orphan::TEXT
                FROM approval WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_male' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT
                FROM faers_male WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_ped' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT
                FROM faers_ped WHERE struct_id = %s
                UNION ALL
                SELECT 'omop_relationship' AS table_name, struct_id::TEXT, concept_name::TEXT, cui_semantic_type::TEXT, relationship_name::TEXT, umls_cui::TEXT
                FROM omop_relationship WHERE struct_id = %s
                UNION ALL
                SELECT 'faers' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT
                FROM faers WHERE struct_id = %s
                UNION ALL
                SELECT 'atc_ddd' AS table_name, struct_id::TEXT, atc_code::TEXT, route::TEXT, ddd::TEXT, unit_type::TEXT
                FROM atc_ddd WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_female' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT
                FROM faers_female WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_ger' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT
                FROM faers_ger WHERE struct_id = %s
                UNION ALL
                SELECT 'struct2obprod' AS table_name, struct_id::TEXT, prod_id::TEXT, strength::TEXT, NULL::TEXT, NULL::TEXT
                FROM struct2obprod WHERE struct_id = %s
                UNION ALL
                SELECT 'pdb' AS table_name, struct_id::TEXT, pdb::TEXT, ligand_id::TEXT, accession::TEXT, pubmed_id::TEXT
                FROM pdb WHERE struct_id = %s
                UNION ALL
                SELECT 'pharma_class' AS table_name, struct_id::TEXT, class_code::TEXT, source::TEXT, name::TEXT, NULL::TEXT
                FROM pharma_class WHERE struct_id = %s
                UNION ALL
                SELECT 'pka' AS table_name, struct_id::TEXT, value::TEXT, pka_type::TEXT, pka_level::TEXT, NULL::TEXT
                FROM pka WHERE struct_id = %s
                UNION ALL
                SELECT 'ob_exclusivity_view' AS table_name, struct_id::TEXT, appl_no::TEXT, trade_name::TEXT, exclusivity_date::TEXT, description::TEXT
                FROM ob_exclusivity_view WHERE struct_id = %s
                UNION ALL
                SELECT 'ob_patent_view' AS table_name, struct_id::TEXT, appl_no::TEXT, trade_name::TEXT, patent_no::TEXT, patent_expire_date::TEXT
                FROM ob_patent_view WHERE struct_id = %s;
            """
            cur.execute(
                query, [struct_id] * 15
            )  # struct_id is used 15 times in the query
            all_data = cur.fetchall()

            # Step 3: Group data by table_name with user-friendly names
            grouped_data = {}
            for row in all_data:
                table_name = row.pop("table_name")
                readable_name = TABLE_NAME_MAPPING.get(
                    table_name, table_name.replace("_", " ").title()
                )
                if readable_name not in grouped_data:
                    grouped_data[readable_name] = []
                grouped_data[readable_name].append(row)

            # Step 4: Fetch additional details (mechanism of action, etc.)
            cur.execute(
                """
                SELECT DISTINCT
                    UPPER(ai.substance_name) AS active_ingredient,
                    td.name AS target_protein,
                    at.action_type,
                    s.mrdef AS mechanism_of_action
                FROM structures s
                LEFT JOIN active_ingredient ai ON s.id = ai.struct_id
                LEFT JOIN act_table_full act ON s.id = act.struct_id
                LEFT JOIN target_dictionary td ON act.target_id = td.id
                LEFT JOIN action_type at ON act.action_type = at.id::VARCHAR
                WHERE UPPER(ai.substance_name) = %s
            """,
                [normalized_drug],
            )
            additional_details = cur.fetchall()

            # Ensure uniqueness using Python (removes duplicates missed by DISTINCT)
            additional_details = list(
                {frozenset(item.items()): item for item in additional_details}.values()
            )

            return render_template(
                "medicine/drug_details.html",
                drug=drug,
                additional_details=additional_details,
                grouped_data=grouped_data,
                struct_id=struct_id,
            )
    except Exception as e:
        print(f"Database error: {str(e)}")
        return render_template(
            "medicine/error.html",
            message="An error occurred while fetching drug details.",
        )


# add patient to theatre lists


@bp.route("/prescriptions/new", methods=["GET", "POST"])
@login_required
def new_prescription():
    if request.method == "POST":
        booking_id = request.form.get("booking_id")
        regimen_id = request.form.get("regimen_id")
        start_date = request.form.get("start_date")
        prescribed_by = request.form.get("prescribed_by")
        notes = request.form.get("notes", "").strip() or None
        drug_ids = request.form.getlist("drug_id")  # List of drug IDs

        # Log form data for debugging
        print(
            f"Form data: booking_id={booking_id}, regimen_id={regimen_id}, start_date={start_date}, prescribed_by={prescribed_by}, notes={notes}, drug_ids={drug_ids}"
        )

        # Validate required fields
        if not booking_id:
            flash("Booking selection is required.", "danger")
            return redirect(url_for("medicine.new_prescription"))
        if not regimen_id:
            flash("Regimen selection is required.", "danger")
            return redirect(url_for("medicine.new_prescription"))
        if not start_date:
            flash("Start date is required.", "danger")
            return redirect(url_for("medicine.new_prescription"))
        if not prescribed_by:
            flash("Prescribed by is required.", "danger")
            return redirect(url_for("medicine.new_prescription"))

        # Validate booking
        booking = OncologyBooking.query.filter_by(
            id=booking_id, purpose="Chemotherapy", status="Scheduled"
        ).first()
        if not booking:
            flash(
                "Selected booking does not exist or is not a scheduled chemotherapy booking.",
                "danger",
            )
            return redirect(url_for("medicine.new_prescription"))

        # Validate patient
        patient = Patient.query.filter_by(patient_id=booking.patient_id).first()
        if not patient:
            flash("Associated patient does not exist.", "danger")
            return redirect(url_for("medicine.new_prescription"))

        # Validate regimen
        regimen = OncologyRegimen.query.filter_by(id=regimen_id).first()
        if not regimen:
            flash("Selected regimen does not exist.", "danger")
            return redirect(url_for("medicine.new_prescription"))

        # Validate drug inputs
        regimen_drugs = RegimenDrugAssociation.query.filter_by(
            regimen_id=regimen_id
        ).all()
        if not regimen_drugs:
            flash("Selected regimen has no associated drugs.", "danger")
            return redirect(url_for("medicine.new_prescription"))

        # Parse start_date
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError as e:
            print(f"Date parsing error: {e}")
            flash("Invalid date format. Use YYYY-MM-DD.", "danger")
            return redirect(url_for("medicine.new_prescription"))

        # Validate and collect drug details
        drug_inputs = {}
        for drug_id in drug_ids:
            dosage = request.form.get(f"dosage_{drug_id}", "").strip() or None
            calculated_dose = (
                request.form.get(f"calculated_dose_{drug_id}", "").strip() or None
            )
            infusion_fluid = (
                request.form.get(f"infusion_fluid_{drug_id}", "").strip() or None
            )
            infusion_time = (
                request.form.get(f"infusion_time_{drug_id}", "").strip() or None
            )
            if not dosage or not calculated_dose:
                flash(
                    f"Dosage and calculated dose are required for drug ID {drug_id}.",
                    "danger",
                )
                return redirect(url_for("medicine.new_prescription"))
            drug_inputs[drug_id] = {
                "dosage": dosage,
                "calculated_dose": calculated_dose,
                "infusion_fluid": infusion_fluid,
                "infusion_time": infusion_time,
            }

        # Create prescription
        try:
            new_prescription = OncoPrescription(
                onco_patient_id=booking.patient_id,
                regimen_id=regimen_id,
                start_date=start_date,
                end_date=None,
                prescribed_by=prescribed_by,
                notes=notes,
                created_at=datetime.now(timezone.utc),
            )
            db.session.add(new_prescription)
            db.session.flush()  # Get prescription ID

            # Add drug details
            for drug_id, details in drug_inputs.items():
                drug = OncologyDrug.query.get(drug_id)
                if not drug:
                    flash(f"Drug ID {drug_id} does not exist.", "danger")
                    db.session.rollback()
                    return redirect(url_for("medicine.new_prescription"))
                drug_detail = PrescriptionDrugDetail(
                    prescription_id=new_prescription.id,
                    drug_id=drug_id,
                    dosage=details["dosage"],
                    calculated_dose=details["calculated_dose"],
                    infusion_fluid=details["infusion_fluid"]
                    or drug.reconstitution_fluid,
                    infusion_time=details["infusion_time"] or drug.infusion_time,
                    created_at=datetime.now(timezone.utc),
                )
                db.session.add(drug_detail)

            db.session.commit()
            flash("Chemotherapy prescription created successfully!", "success")
            return redirect(
                url_for("medicine.list_prescriptions", patient_id=booking.patient_id)
            )
        except Exception as e:
            db.session.rollback()
            print(f"Error creating prescription: {e}")
            flash("An error occurred while creating the prescription.", "danger")
            return redirect(url_for("medicine.new_prescription"))

    # GET: Render form
    bookings = OncologyBooking.query.filter_by(
        purpose="Chemotherapy", status="Scheduled"
    ).all()
    regimens = OncologyRegimen.query.all()

    # Prepare booking details
    booking_details = [
        {
            "id": booking.id,
            "patient_name": booking.patient.name,
            "patient_id": booking.patient_id,
            "booking_date": booking.booking_date.strftime("%Y-%m-%d"),
        }
        for booking in bookings
    ]

    # Prepare regimen drugs
    regimen_drugs = {}
    for regimen in regimens:
        drugs = (
            RegimenDrugAssociation.query.filter_by(regimen_id=regimen.id)
            .order_by(RegimenDrugAssociation.sequence)
            .all()
        )
        regimen_drugs[regimen.id] = [
            {
                "id": drug.drug.id,
                "name": drug.drug.name,
                "dose": drug.dose or "N/A",
                "administration_route": drug.administration_route or "N/A",
                "administration_schedule": drug.administration_schedule or "N/A",
                "reconstitution_fluid": drug.drug.reconstitution_fluid or "N/A",
                "infusion_time": drug.drug.infusion_time or "N/A",
            }
            for drug in drugs
        ]

    return render_template(
        "medicine/oncology/new_prescription.html",
        bookings=booking_details,
        regimens=regimens,
        regimen_drugs=regimen_drugs,
        no_bookings=len(booking_details) == 0,
        no_regimens=len(regimens) == 0,
    )


@bp.route("/prescriptions/<patient_id>", methods=["GET"])
@login_required
def list_prescriptions(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash("Patient does not exist.", "danger")
        return redirect(url_for("medicine.new_prescription"))

    prescriptions = OncoPrescription.query.filter_by(onco_patient_id=patient_id).all()
    prescription_details = []

    for prescription in prescriptions:
        regimen = OncologyRegimen.query.get(prescription.regimen_id)
        drug_details = PrescriptionDrugDetail.query.filter_by(
            prescription_id=prescription.id
        ).all()
        drugs = [
            {
                "name": OncologyDrug.query.get(detail.drug_id).name,
                "dosage": detail.dosage,
                "calculated_dose": detail.calculated_dose,
                "infusion_fluid": detail.infusion_fluid,
                "infusion_time": detail.infusion_time,
            }
            for detail in drug_details
        ]
        prescription_details.append(
            {
                "id": prescription.id,
                "regimen_name": regimen.name if regimen else "Unknown Regimen",
                "start_date": prescription.start_date.strftime("%Y-%m-%d"),
                "prescribed_by": prescription.prescribed_by,
                "notes": prescription.notes,
                "drugs": drugs,
                "created_at": prescription.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return render_template(
        "medicine/oncology/list_prescriptions.html",
        patient=patient,
        prescriptions=prescription_details,
    )


# New route to list all prescriptions
@bp.route("/prescriptions", methods=["GET"])
@login_required
def all_prescriptions():
    # Query all prescriptions, joining with patients and regimens for display
    prescriptions = (
        db.session.query(OncoPrescription, Patient, OncologyRegimen)
        .join(Patient, OncoPrescription.onco_patient_id == Patient.patient_id)
        .join(OncologyRegimen, OncoPrescription.regimen_id == OncologyRegimen.id)
        .all()
    )

    prescription_details = []
    for prescription, patient, regimen in prescriptions:
        drug_details = PrescriptionDrugDetail.query.filter_by(
            prescription_id=prescription.id
        ).all()
        drugs = [
            {
                "name": OncologyDrug.query.get(detail.drug_id).name,
                "dosage": detail.dosage,
                "calculated_dose": detail.calculated_dose,
                "infusion_fluid": detail.infusion_fluid,
                "infusion_time": detail.infusion_time,
            }
            for detail in drug_details
        ]
        prescription_details.append(
            {
                "id": prescription.id,
                "patient_id": patient.patient_id,
                "patient_name": patient.name,
                "regimen_name": regimen.name,
                "start_date": prescription.start_date.strftime("%Y-%m-%d"),
                "prescribed_by": prescription.prescribed_by,
                "notes": prescription.notes,
                "drugs": drugs,
                "created_at": prescription.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return render_template(
        "medicine/oncology/all_prescriptions.html", prescriptions=prescription_details
    )


# cancers
