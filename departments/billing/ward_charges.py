"""
departments/billing/ward_charges.py

T3.5: Midnight cron job to post daily room & board charges
for all active IPD encounters.
"""
from datetime import datetime, timezone

from sqlalchemy import and_

from departments.models.billing import Invoice, InvoiceLineItem
from departments.models.encounter import Encounter
from departments.models.medicine import AdmittedPatient, Ward
from extensions import db


def post_daily_ward_charges():
    """
    Finds all active IPD encounters, calculates the daily ward rate,
    and creates an InvoiceLineItem for today's room & board.
    """
    today = datetime.now(timezone.utc).date()

    # 1. Find all active IPD encounters
    active_ipds = db.session.query(Encounter).filter(
        and_(
            Encounter.encounter_type == "IPD",
            Encounter.stage == "ADMITTED"
        )
    ).all()

    charges_posted = 0

    for enc in active_ipds:
        # 2. Find the active admission for this patient (linked via patient_id)
        admission = db.session.query(AdmittedPatient).filter(
            and_(
                AdmittedPatient.patient_id == enc.patient_id,
                AdmittedPatient.discharged_on.is_(None)
            )
        ).first()

        if not admission or not admission.ward_id:
            continue

        ward = db.session.query(Ward).get(admission.ward_id)
        if not ward or ward.daily_charge is None:
            continue

        # 3. Idempotency check: Ensure we haven't already billed for this encounter today
        existing_charge = db.session.query(InvoiceLineItem).filter(
            and_(
                InvoiceLineItem.encounter_id == enc.id,
                InvoiceLineItem.description.like(f"Ward Charge - {today}%")
            )
        ).first()

        if existing_charge:
            continue

        # 4. Get or create the patient's open invoice
        invoice = db.session.query(Invoice).filter_by(
            patient_id=enc.patient_id,
            status="OPEN"
        ).first()

        if not invoice:
            invoice = Invoice(patient_id=enc.patient_id, status="OPEN", created_at=datetime.now(timezone.utc))
            db.session.add(invoice)
            db.session.flush()

        # 5. Create the line item
        line_item = InvoiceLineItem(
            invoice_id=invoice.id,
            encounter_id=enc.id,
            description=f"Ward Charge - {ward.name} - {today.strftime('%Y-%m-%d')}",
            amount=ward.daily_charge,
            created_at=datetime.now(timezone.utc)
        )
        db.session.add(line_item)
        charges_posted += 1

    db.session.commit()
    print(f"✅ T3.5: Posted {charges_posted} daily ward charges for {today}.")
    return charges_posted
