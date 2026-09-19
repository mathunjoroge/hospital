import os
import uuid
from datetime import datetime, timezone

from flask import flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required
from psycopg2.extras import RealDictCursor

from departments.clinical_safety.engine import ClinicalSafetyEngine
from departments.medicine.orders import fetch_drugs_data
from departments.models.compliance import has_consent
from departments.models.medicine import (
    AdmittedPatient,
    Medicine,
    OncologyBooking,
    OncologyDrug,
    OncologyRegimen,
    OncoPatient,
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
from departments.shared.encounter_utils import active_encounter
from extensions import db

from . import bp
from .oncology import (
    CLINICAL_READ_ROLES,
    CLINICAL_WRITE_ROLES,
    _commit_with_audit,
    _find_patient,
)

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

        # Get patient record — exact match only. A fuzzy ilike on id/name
        # could save the prescription onto a different patient than the one
        # the clinician selected ("P1" resolving to "P10").
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f"Patient with ID {patient_id} not found in the system!", "error")
            return redirect(url_for("medicine.index"))

        # Per-patient prescription draft key. A single shared session key was
        # clobbered when a clinician worked on two patients in separate tabs.
        draft_key = f"prescription_id_{patient.patient_id}"
        if draft_key not in session:
            session[draft_key] = str(uuid.uuid4())
        prescription_id = session[draft_key]

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

            # Save prescriptions — scoped to the active encounter for this
            # patient's visit.
            encounter = active_encounter(patient.patient_id)
            for drug_id in drugs_selected:
                prescribed = PrescribedMedicine(
                    patient_id=patient.patient_id,
                    encounter_id=encounter.id if encounter else None,
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

            except Exception:  # noqa: BLE001
                db.session.rollback()
                logger.exception("Database commit failed in prescribe_drugs")
                flash("Something went wrong. Please try again.", "error")
                return redirect(
                    url_for(
                        "medicine.prescribe_drugs", patient_id=patient_id, dept=dept
                    )
                )

        # Consent check.
        #
        # `patient_id` here is the business identifier ("P0001"), not the
        # numeric PK. This block previously did int(patient_id), which raised
        # for every non-numeric ID, left patient_int_id as None, and silently
        # skipped BOTH this consent gate and the CDS check below. Resolve the
        # two identifiers explicitly instead of casting.
        patient_int_id = patient.id if patient else None
        consent_status = (
            "ACTIVE"
            if (patient and has_consent(patient.patient_id, "TREATMENT"))
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

        prescribed_medicine = db.session.get(PrescribedMedicine, medicine_id)
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
        if prescribed_medicine.status == 1:
            flash(
                "This prescription has already been dispensed by pharmacy "
                "and can no longer be deleted.",
                "error",
            )
            return redirect(
                url_for(
                    "medicine.prescribe_drugs",
                    patient_id=patient_id,
                    prescription_id=prescription_id,
                    dept=dept,
                )
            )
        db.session.delete(prescribed_medicine)
        db.session.commit()
        flash("Prescribed medicine deleted successfully!", "success")

    except Exception as e:  # noqa: BLE001
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

        # ✅ Clear the per-patient prescription draft (matches the per-patient
        # key used by prescribe_drugs).
        session.pop(f"prescription_id_{patient_id}", None)

        flash("Prescription finalized and saved successfully.", "success")

        # Redirect based on dept value
        if dept == "1":
            return redirect(url_for("medicine.ward_rounds"))
        else:
            return redirect(url_for("medicine.index"))

    except Exception as e:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        logger.exception("Error serving medicine edit form")
        return "Error loading edit form."


@bp.route("/drugs-ref/search", methods=["GET"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def drugs_ref():
    """Drugs reference route with search functionality."""
    search_query = request.args.get("search", "").strip()
    drugs_data = fetch_drugs_data(search_query)

    return render_template(
        "medicine/drugs_ref.html", drugs_data=drugs_data, search_query=search_query
    )


@bp.route("/drugs-ref/details/<drug>", methods=["GET"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def drug_details(drug: str):
    """Fetch and display detailed information about a specific active ingredient."""
    normalized_drug = drug.strip().upper()

    try:
        with get_db_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
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

            # Tier 1: Exact match on active_ingredient.substance_name
            cur.execute(
                """
                SELECT DISTINCT struct_id
                FROM active_ingredient
                WHERE UPPER(substance_name) = %s
                LIMIT 1
            """,
                [normalized_drug],
            )
            result = cur.fetchone()

            # Tier 2: Search product table for generic_name or product_name match
            if not result:
                cur.execute(
                    """
                    SELECT DISTINCT s.struct_id
                    FROM product p
                    JOIN struct2obprod s ON p.prod_id = s.prod_id
                    WHERE UPPER(p.generic_name) = %s OR UPPER(p.product_name) = %s
                    LIMIT 1
                """,
                    [normalized_drug, normalized_drug],
                )
                result = cur.fetchone()

            # Tier 3: Search active_ingredient with ILIKE or partial match
            if not result:
                cur.execute(
                    """
                    SELECT DISTINCT struct_id
                    FROM active_ingredient
                    WHERE UPPER(substance_name) ILIKE %s
                    LIMIT 1
                """,
                    [f"%{normalized_drug}%"],
                )
                result = cur.fetchone()

            # Tier 4: For compound/multi-ingredient strings (comma or 'and' separated),
            # extract candidate ingredients and find the first matching active ingredient struct_id
            if not result and ("," in drug or " and " in drug.lower() or "/" in drug):
                raw_parts = [
                    p.strip().upper()
                    for item in drug.replace(" and ", ",").split(",")
                    for p in item.split("/")
                    if p.strip()
                ]
                for part in raw_parts:
                    if len(part) < 3:
                        continue
                    cur.execute(
                        """
                        SELECT DISTINCT struct_id
                        FROM active_ingredient
                        WHERE UPPER(substance_name) = %s OR UPPER(substance_name) ILIKE %s
                        LIMIT 1
                    """,
                        [part, f"%{part}%"],
                    )
                    sub_match = cur.fetchone()
                    if sub_match:
                        result = sub_match
                        break

            if result:
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
                )
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
    except Exception:  # noqa: BLE001
        logger.exception("Database error fetching DrugCentral drug details")

    # Local hospital database fallback
    from departments.models.medicine import Medicine, OncologyDrug
    from departments.models.pharmacy import Drug as PharmDrug

    onco = OncologyDrug.query.filter(OncologyDrug.name.ilike(drug)).first()
    med = Medicine.query.filter(
        (Medicine.generic_name.ilike(drug)) | (Medicine.brand_name.ilike(drug))
    ).first()
    pharm = PharmDrug.query.filter(PharmDrug.generic_name.ilike(drug)).first()

    if onco or med or pharm:
        name = onco.name if onco else (med.generic_name if med else pharm.generic_name)
        form = (
            onco.dosage_form
            if onco
            else (med.dosage if med else pharm.dosage_form)
        )
        strength = onco.strength if onco else (pharm.strength if pharm else "Standard")
        moa = (
            onco.mechanism_of_action
            if (onco and onco.mechanism_of_action)
            else "Refer to clinical reference manual."
        )

        grouped_data = {}
        if onco and onco.side_effects:
            grouped_data["Side Effects"] = [{"description": onco.side_effects}]
        if onco and onco.therapeutic_class:
            grouped_data["Pharmacological Class"] = [
                {"class_name": onco.therapeutic_class, "category": "Oncology"}
            ]

        additional_details = [
            {
                "active_ingredient": name,
                "target_protein": form or "N/A",
                "action_type": strength or "N/A",
                "mechanism_of_action": moa,
            }
        ]
        return render_template(
            "medicine/drug_details.html",
            drug=name,
            additional_details=additional_details,
            grouped_data=grouped_data,
            struct_id=None,
            is_local=True,
        )

    return render_template(
        "medicine/error.html",
        message=f"No details found for {drug}.",
    )


# add patient to theatre lists


def _prescription_drug_rows(prescription_id):
    """Drug lines of one prescription, tolerating a drug that has since been removed."""
    rows = []
    for detail in PrescriptionDrugDetail.query.filter_by(
        prescription_id=prescription_id
    ).all():
        drug = db.session.get(OncologyDrug, detail.drug_id)
        rows.append(
            {
                "name": drug.name if drug else "Unknown drug",
                "dosage": detail.dosage,
                "calculated_dose": detail.calculated_dose,
                "infusion_fluid": detail.infusion_fluid,
                "infusion_time": detail.infusion_time,
            }
        )
    return rows


@bp.route("/prescriptions/new", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def new_prescription():
    """
    Legacy free-text chemotherapy prescription form.

    NOTE: dosage / calculated_dose are still clinician-typed text here; the
    server-validated path is the chemotherapy builder (BSA + lifetime caps).
    """
    if request.method == "POST":
        booking_id = request.form.get("booking_id")
        regimen_id = request.form.get("regimen_id")
        start_date = request.form.get("start_date")
        notes = request.form.get("notes", "").strip() or None
        drug_ids = request.form.getlist("drug_id")  # List of drug IDs

        # The prescriber is the authenticated user. It used to be a free-text
        # form field, i.e. anyone could prescribe "as" anyone else.
        prescribed_by = (
            getattr(current_user, "full_name", None) or current_user.username
        )[:100]

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
        if not drug_ids:
            flash("Select at least one drug for the prescription.", "danger")
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

        # Validate patient (exact patient number)
        patient = Patient.query.filter_by(patient_id=booking.patient_id).first()
        if not patient:
            flash("Associated patient does not exist.", "danger")
            return redirect(url_for("medicine.new_prescription"))

        # onco_prescriptions.onco_patient_id is an INTEGER FK to onco_patients.id,
        # so resolve the patient's enrolment row; never store the patient number there.
        onco_patient = (
            OncoPatient.query.filter_by(patient_id=patient.patient_id, status="Active")
            .order_by(OncoPatient.id.desc())
            .first()
        )
        if not onco_patient:
            flash(
                "Patient is not enrolled in active oncology care. Enrol the patient first.",
                "danger",
            )
            return redirect(url_for("medicine.new_prescription"))

        # Validate regimen (and that it is still in use)
        regimen = OncologyRegimen.query.filter_by(id=regimen_id).first()
        if not regimen:
            flash("Selected regimen does not exist.", "danger")
            return redirect(url_for("medicine.new_prescription"))
        if (regimen.status or "Active") != "Active":
            flash(
                f"Regimen '{regimen.name}' is {regimen.status} and cannot be prescribed.",
                "danger",
            )
            return redirect(url_for("medicine.new_prescription"))

        # Validate drug inputs: each drug must be a member of the chosen regimen.
        regimen_drugs = RegimenDrugAssociation.query.filter_by(
            regimen_id=regimen.id
        ).all()
        if not regimen_drugs:
            flash("Selected regimen has no associated drugs.", "danger")
            return redirect(url_for("medicine.new_prescription"))
        regimen_drug_ids = {rd.drug_id for rd in regimen_drugs}

        # Parse start_date
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()  # noqa: DTZ007
        except ValueError:
            flash("Invalid date format. Use YYYY-MM-DD.", "danger")
            return redirect(url_for("medicine.new_prescription"))

        # Validate and collect drug details
        drug_inputs = {}
        for raw_drug_id in drug_ids:
            try:
                drug_id = int(raw_drug_id)
            except (TypeError, ValueError):
                flash(f"Invalid drug ID {raw_drug_id}.", "danger")
                return redirect(url_for("medicine.new_prescription"))
            if drug_id not in regimen_drug_ids:
                flash(
                    f"Drug ID {drug_id} is not part of regimen '{regimen.name}'.",
                    "danger",
                )
                return redirect(url_for("medicine.new_prescription"))
            dosage = request.form.get(f"dosage_{raw_drug_id}", "").strip() or None
            calculated_dose = (
                request.form.get(f"calculated_dose_{raw_drug_id}", "").strip() or None
            )
            infusion_fluid = (
                request.form.get(f"infusion_fluid_{raw_drug_id}", "").strip() or None
            )
            infusion_time = (
                request.form.get(f"infusion_time_{raw_drug_id}", "").strip() or None
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
                onco_patient_id=onco_patient.id,
                regimen_id=regimen.id,
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
                drug = db.session.get(OncologyDrug, drug_id)
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

            if not _commit_with_audit(
                "CHEMO_PRESCRIPTION_CREATED",
                "OncoPrescription",
                new_prescription.id,
                {
                    "patient_id": patient.patient_id,
                    "regimen_id": regimen.id,
                    "drug_ids": sorted(drug_inputs),
                },
            ):
                flash(
                    "Audit trail unavailable; the prescription was NOT saved.", "danger"
                )
                return redirect(url_for("medicine.new_prescription"))
            flash("Chemotherapy prescription created successfully!", "success")
            return redirect(
                url_for("medicine.list_prescriptions", patient_id=patient.patient_id)
            )
        except Exception:  # noqa: BLE001
            db.session.rollback()
            logger.exception("Error creating chemotherapy prescription")
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
@roles_required(*CLINICAL_READ_ROLES)
def list_prescriptions(patient_id):
    # Exact match only. A fuzzy id/name match used to show one patient's header
    # above prescriptions that were looked up under a different identifier.
    patient = _find_patient(patient_id)
    if not patient:
        flash("Patient does not exist.", "danger")
        return redirect(url_for("medicine.new_prescription"))

    onco_ids = [
        row.id for row in OncoPatient.query.filter_by(patient_id=patient.patient_id)
    ]
    prescriptions = (
        OncoPrescription.query.filter(OncoPrescription.onco_patient_id.in_(onco_ids))
        .order_by(OncoPrescription.created_at.desc())
        .all()
        if onco_ids
        else []
    )
    prescription_details = []

    for prescription in prescriptions:
        regimen = db.session.get(OncologyRegimen, prescription.regimen_id)
        prescription_details.append(
            {
                "id": prescription.id,
                "regimen_name": regimen.name if regimen else "Unknown Regimen",
                "start_date": prescription.start_date.strftime("%Y-%m-%d"),
                "prescribed_by": prescription.prescribed_by,
                "notes": prescription.notes,
                "drugs": _prescription_drug_rows(prescription.id),
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
@roles_required(*CLINICAL_READ_ROLES)
def all_prescriptions():
    # Query all prescriptions, joining with patients and regimens for display.
    # OncoPrescription -> OncoPatient (integer id) -> Patient (patient number).
    prescriptions = (
        db.session.query(OncoPrescription, Patient, OncologyRegimen)
        .join(OncoPatient, OncoPrescription.onco_patient_id == OncoPatient.id)
        .join(Patient, OncoPatient.patient_id == Patient.patient_id)
        .join(OncologyRegimen, OncoPrescription.regimen_id == OncologyRegimen.id)
        .order_by(OncoPrescription.created_at.desc())
        .all()
    )

    prescription_details = []
    for prescription, patient, regimen in prescriptions:
        prescription_details.append(
            {
                "id": prescription.id,
                "patient_id": patient.patient_id,
                "patient_name": patient.name,
                "regimen_name": regimen.name,
                "start_date": prescription.start_date.strftime("%Y-%m-%d"),
                "prescribed_by": prescription.prescribed_by,
                "notes": prescription.notes,
                "drugs": _prescription_drug_rows(prescription.id),
                "created_at": prescription.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return render_template(
        "medicine/oncology/all_prescriptions.html", prescriptions=prescription_details
    )


# cancers
