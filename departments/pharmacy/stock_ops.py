"""
departments/pharmacy/stock_ops.py
─────────────────────────────────
Extracted stock management, FEFO dispensing processing, and patient history routes
from dispensing.py to keep file sizes modular and clean.
"""

import json
import logging
from datetime import datetime, timezone

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy.orm import joinedload

from departments.models.medicine import PrescribedMedicine
from departments.models.pharmacy import Batch, DispensedDrug, Drug
from departments.models.records import Patient
from departments.rbac import roles_required
from extensions import db

from . import bp

logger = logging.getLogger(__name__)


@bp.route("/remove_dispensed/<int:dispense_id>", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def remove_dispensed(dispense_id):
    """Remove a dispensed drug entry."""
    try:
        dispensed_drug = DispensedDrug.query.get(dispense_id)
        if not dispensed_drug:
            flash(f"Dispensed drug with ID {dispense_id} does not exist!", "error")
            return redirect(
                url_for(
                    "pharmacy.dispense_prescription",
                    prescription_id=request.form.get("prescription_id"),
                )
            )

        batch = (
            Batch.query.get(dispensed_drug.batch_id)
            if dispensed_drug.batch_id
            else None
        )
        if batch:
            batch.quantity_in_stock += dispensed_drug.quantity_dispensed
            db.session.add(batch)

        db.session.delete(dispensed_drug)
        db.session.commit()

        flash(
            f"{dispensed_drug.drug.generic_name} removed from dispensing list!",
            "success",
        )
        return redirect(
            url_for(
                "pharmacy.dispense_prescription",
                prescription_id=dispensed_drug.prescription_id,
            )
        )

    except Exception as e:  # noqa: BLE001
        flash("Something went wrong. Please try again.", "error")
        logger.error(f"Error in pharmacy.remove_dispensed: {e}")
        db.session.rollback()
        return redirect(
            url_for(
                "pharmacy.dispense_prescription",
                prescription_id=request.form.get("prescription_id"),
            )
        )


@bp.route("/dispense/process/<string:prescription_id>", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def process_dispense(prescription_id):
    """
    Handles dispensing of drugs, updates stock, and renders the dispense prescription page with dispensed drug details.
    """
    try:
        drug_id = request.form.get("drug_id")
        batch_id = request.form.get("batch_id")
        quantity_dispensed = request.form.get("quantity_dispensed")

        if not drug_id or not batch_id or not quantity_dispensed:
            flash(
                "Invalid input! Please select a drug and specify its quantity.", "error"
            )
            return redirect(
                url_for(
                    "pharmacy.dispense_prescription", prescription_id=prescription_id
                )
            )

        try:
            quantity_dispensed = int(quantity_dispensed)
            if quantity_dispensed <= 0:
                raise ValueError("Quantity must be greater than zero.")
        except ValueError:
            flash("Invalid quantity! Enter a positive number.", "error")
            return redirect(
                url_for(
                    "pharmacy.dispense_prescription", prescription_id=prescription_id
                )
            )

        drug = Drug.query.get(drug_id)
        if not drug:
            flash(f"Drug with ID {drug_id} does not exist!", "error")
            return redirect(
                url_for(
                    "pharmacy.dispense_prescription", prescription_id=prescription_id
                )
            )

        batch = Batch.query.filter_by(id=batch_id, drug_id=drug.id).first()
        if not batch:
            flash(
                f"Batch ID {batch_id} does not exist for {drug.generic_name}.", "error"
            )
            return redirect(
                url_for(
                    "pharmacy.dispense_prescription", prescription_id=prescription_id
                )
            )

        if batch.quantity_in_stock < quantity_dispensed:
            flash(
                f"Insufficient stock for {drug.generic_name}. Available: {batch.quantity_in_stock}",
                "error",
            )
            return redirect(
                url_for(
                    "pharmacy.dispense_prescription", prescription_id=prescription_id
                )
            )

        prescribed_medicine = PrescribedMedicine.query.filter_by(
            prescription_id=prescription_id
        ).first()
        if not prescribed_medicine:
            flash("Prescription not found.", "error")
            return redirect(url_for("pharmacy.index"))

        patient_id = prescribed_medicine.patient_id

        new_dispensed_drug = DispensedDrug(
            drug_id=drug.id,
            batch_id=batch.id,
            patient_id=patient_id,
            prescription_id=prescription_id,
            quantity_dispensed=quantity_dispensed,
            date_dispensed=datetime.now(timezone.utc).date(),
            status="Pending",
        )
        db.session.add(new_dispensed_drug)

        batch.quantity_in_stock -= quantity_dispensed
        db.session.add(batch)

        db.session.commit()
        flash(
            f"{drug.generic_name} ({quantity_dispensed} units) dispensed successfully!",
            "success",
        )

        dispensed_drugs = (
            db.session.query(
                Drug.generic_name,
                Drug.brand_name,
                Drug.dosage_form,
                Drug.strength,
                Drug.selling_price,
                DispensedDrug.id,
                DispensedDrug.batch_id,
                DispensedDrug.quantity_dispensed,
                (Drug.selling_price * DispensedDrug.quantity_dispensed).label("total"),
            )
            .join(DispensedDrug, DispensedDrug.drug_id == Drug.id)
            .filter(DispensedDrug.prescription_id == prescription_id)
            .all()
        )

        dispensed_drugs_list = [
            {
                "generic_name": d.generic_name,
                "id": d.id,
                "brand_name": d.brand_name,
                "dosage_form": d.dosage_form,
                "strength": d.strength,
                "selling_price": d.selling_price,
                "batch_id": d.batch_id,
                "quantity_dispensed": d.quantity_dispensed,
                "total": d.total,
            }
            for d in dispensed_drugs
        ]

        prescribed_medicines = PrescribedMedicine.query.filter_by(
            prescription_id=prescription_id
        ).all()
        drug_batches = {
            b.drug_id: b
            for b in Batch.query.filter(
                Batch.drug_id.in_([m.medicine_id for m in prescribed_medicines])
            ).all()
        }

        return render_template(
            "pharmacy/dispense_prescription.html",
            prescription_id=prescription_id,
            prescribed_medicines=prescribed_medicines,
            drug_batches=drug_batches,
            drugs=Drug.query.all(),
            dispensed_drugs=dispensed_drugs_list,
        )

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        logger.error(f"Error in pharmacy.process_dispense: {e}")
        return redirect(url_for("pharmacy.index"))


@bp.route("/patient_history", methods=["GET", "POST"])
@login_required
def patient_history():
    """API endpoint to retrieve patient history by patient_id with clinical data."""
    from departments.models.billing import Billing
    from departments.models.imaging import ImagingResult
    from departments.models.laboratory import LabResult, LabResultTemplate
    from departments.models.medicine import (
        AdmittedPatient,
        RequestedImage,
        RequestedLab,
        SOAPNote,
    )
    from departments.models.nursing import Vitals
    from departments.models.records import ClinicBooking

    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            patient_id = data.get("patient_id", "").strip()
        else:
            patient_id = request.form.get("patient_id", "").strip()

        if not patient_id:
            if request.is_json:
                return jsonify({"error": "Patient ID is required"}), 400
            flash("Please provide a patient ID.", "error")
            return render_template(
                "pharmacy/pharmacy_dashboard.html", patient_history_form=True
            )

        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            if request.is_json:
                return jsonify({"error": f"No patient found with ID {patient_id}"}), 404
            flash(f"No patient found with ID {patient_id}.", "error")
            return render_template(
                "pharmacy/pharmacy_dashboard.html", patient_history_form=True
            )

        prescribed_meds = (
            PrescribedMedicine.query.filter_by(patient_id=patient_id)
            .options(joinedload(PrescribedMedicine.medicine))
            .all()
        )
        dispensed_drugs = (
            DispensedDrug.query.filter_by(patient_id=patient_id)
            .options(joinedload(DispensedDrug.drug), joinedload(DispensedDrug.batch))
            .all()
        )
        requested_labs = (
            RequestedLab.query.filter_by(patient_id=patient_id)
            .options(joinedload(RequestedLab.lab_test))
            .all()
        )
        lab_results = (
            LabResult.query.filter_by(patient_id=patient_id)
            .options(joinedload(LabResult.lab_test))
            .all()
        )

        test_presentations = {}
        for result in lab_results:
            results_dict = {}
            try:
                results_dict = json.loads(result.result) if result.result else {}
            except json.JSONDecodeError:
                flash(
                    f"Invalid result format for result ID {result.result_id}.",
                    "warning",
                )

            parameters = LabResultTemplate.query.filter_by(
                test_id=result.lab_test_id
            ).all()
            test_presentation = []
            for param in parameters:
                result_value = results_dict.get(str(param.id))
                try:
                    result_value_float = (
                        float(result_value) if result_value is not None else None
                    )
                except (ValueError, TypeError):
                    result_value_float = None

                status = (
                    "Invalid Result"
                    if result_value_float is None
                    else "Low"
                    if result_value_float < param.normal_range_low
                    else "High"
                    if result_value_float > param.normal_range_high
                    else "Normal"
                )

                test_presentation.append(
                    {
                        "parameter_name": param.parameter_name,
                        "normal_range_low": param.normal_range_low,
                        "normal_range_high": param.normal_range_high,
                        "unit": param.unit,
                        "result": result_value if result_value is not None else "N/A",
                        "status": status,
                    }
                )

            test_presentations[result.result_id] = {
                "test_name": result.lab_test.test_name
                if result.lab_test
                else "Unknown Test",
                "test_date": result.test_date,
                "result_notes": result.result_notes,
                "parameters": test_presentation,
            }

        requested_images = (
            RequestedImage.query.filter_by(patient_id=patient_id)
            .options(joinedload(RequestedImage.imaging))
            .all()
        )
        imaging_results = (
            ImagingResult.query.filter_by(patient_id=patient_id)
            .options(joinedload(ImagingResult.imaging))
            .all()
        )
        vitals = Vitals.query.filter_by(patient_id=patient_id).all()
        soap_notes = SOAPNote.query.filter_by(patient_id=patient_id).all()
        clinic_bookings = (
            ClinicBooking.query.filter_by(patient_id=patient_id)
            .options(joinedload(ClinicBooking.clinic))
            .all()
        )
        admissions = (
            AdmittedPatient.query.filter_by(patient_id=patient_id)
            .options(joinedload(AdmittedPatient.ward))
            .all()
        )
        billing = (
            Billing.query.filter_by(patient_id=patient_id)
            .options(joinedload(Billing.charge))
            .all()
        )

        history_data = {
            "patient": {
                "id": patient.id,
                "patient_id": patient.patient_id,
                "name": patient.name,
                "sex": patient.sex,
                "contact": patient.contact,
                "date_registered": patient.date_registered.strftime("%Y-%m-%d %H:%M:%S")
                if patient.date_registered
                else None,
            },
            "prescribed_meds": [
                {
                    "id": m.id,
                    "medicine": m.medicine.generic_name if m.medicine else "N/A",
                    "dosage": m.dosage,
                    "frequency": m.frequency,
                }
                for m in prescribed_meds
            ],
            "dispensed_drugs": [
                {
                    "id": d.id,
                    "drug": d.drug.generic_name if d.drug else "N/A",
                    "quantity": d.quantity_dispensed,
                    "date": d.date_dispensed.strftime("%Y-%m-%d"),
                }
                for d in dispensed_drugs
            ],
            "requested_labs": [
                {
                    "id": req_lab.id,
                    "test_name": req_lab.lab_test.test_name
                    if req_lab.lab_test
                    else "N/A",
                    "status": req_lab.status,
                }
                for req_lab in requested_labs
            ],
            "lab_results": test_presentations,
            "requested_images": [
                {
                    "id": i.id,
                    "type": i.imaging.imaging_type if i.imaging else "N/A",
                    "status": i.status,
                }
                for i in requested_images
            ],
            "imaging_results": [
                {
                    "id": ir.id,
                    "type": ir.imaging.imaging_type if ir.imaging else "N/A",
                    "findings": ir.ai_findings,
                }
                for ir in imaging_results
            ],
            "vitals": [
                {
                    "temp": v.temperature,
                    "bp": f"{v.blood_pressure_systolic}/{v.blood_pressure_diastolic}",
                    "pulse": v.pulse,
                }
                for v in vitals
            ],
            "soap_notes": [
                {
                    "id": s.id,
                    "assessment": s.assessment,
                    "recommendation": s.recommendation,
                }
                for s in soap_notes
            ],
            "clinic_bookings": [
                {
                    "id": cb.id,
                    "clinic": cb.clinic.name if cb.clinic else "N/A",
                    "date": cb.clinic_date.strftime("%Y-%m-%d"),
                }
                for cb in clinic_bookings
            ],
            "admissions": [
                {
                    "id": a.id,
                    "ward": a.ward.name if a.ward else "N/A",
                    "admitted_on": a.admitted_on.strftime("%Y-%m-%d"),
                }
                for a in admissions
            ],
            "billing": [
                {
                    "id": b.id,
                    "charge": b.charge.name if b.charge else "N/A",
                    "cost": float(b.total_cost),
                    "status": b.status,
                }
                for b in billing
            ],
        }

        if request.is_json:
            return jsonify(history_data), 200

        return render_template(
            "pharmacy/pharmacy_dashboard.html",
            patient=patient,
            prescribed_meds=prescribed_meds,
            dispensed_drugs=dispensed_drugs,
            requested_labs=requested_labs,
            lab_results=test_presentations,
            requested_images=requested_images,
            imaging_results=imaging_results,
            vitals=vitals,
            soap_notes=soap_notes,
            clinic_bookings=clinic_bookings,
            admissions=admissions,
            billing=billing,
            patient_history_form=True,
        )

    return render_template(
        "pharmacy/pharmacy_dashboard.html", patient_history_form=True
    )
