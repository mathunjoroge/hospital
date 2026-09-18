"""
departments/pharmacy/controlled_drugs_routes.py
────────────────────────────────────────────────
Frontend + API routes for the Controlled Drug Register.

Provides:
  GET  /pharmacy/controlled-drugs                – HTML console
  GET  /pharmacy/controlled-drugs/api/ledger     – JSON ledger (AJAX)
  POST /pharmacy/controlled-drugs/api/dispense   – Dual-signature dispense
  GET  /pharmacy/controlled-drugs/api/reconcile/<date>/<shift>  – Shift list
  POST /pharmacy/controlled-drugs/api/reconcile  – Submit shift reconciliation
"""

import logging
from datetime import date

from flask import jsonify, render_template, request
from flask_login import current_user, login_required

from departments.models.controlled_drugs import (
    ControlledDrugBalance,
    ControlledDrugDispense,
    ShiftReconciliation,
)
from departments.models.pharmacy import Drug
from departments.models.user import User
from departments.pharmacy.controlled_drugs_service import (
    ControlledDrugError,
    dispense_controlled_drug,
    submit_shift_reconciliation,
)
from departments.rbac import roles_required

from . import bp

logger = logging.getLogger(__name__)


@bp.route("/controlled-drugs", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin", "doctor", "nursing", "medicine")
def controlled_drug_register():
    """
    GET /pharmacy/controlled-drugs
    Render the Controlled Drug Register console.
    """
    # All controlled substances
    controlled_drugs = (
        Drug.query.filter_by(is_controlled=True).order_by(Drug.generic_name).all()
    )

    # Build balance summary for each controlled drug
    drug_summaries = []
    for drug in controlled_drugs:
        last_balance = (
            ControlledDrugBalance.query.filter_by(drug_id=drug.id)
            .order_by(ControlledDrugBalance.recorded_at.desc())
            .first()
        )

        current_balance = last_balance.balance_after if last_balance else 0.0

        drug_summaries.append(
            {
                "id": drug.id,
                "generic_name": drug.generic_name,
                "brand_name": drug.brand_name or "",
                "schedule_class": drug.schedule_class or "Schedule IV",
                "dosage_form": drug.dosage_form,
                "strength": drug.strength,
                "current_balance": current_balance,
                "last_updated": last_balance.recorded_at.strftime("%Y-%m-%d %H:%M")
                if last_balance
                else "—",
            }
        )

    # Recent dispenses (last 50)
    recent_dispenses = (
        ControlledDrugDispense.query.order_by(
            ControlledDrugDispense.dispense_datetime.desc()
        )
        .limit(50)
        .all()
    )

    # Recent reconciliations (last 20)
    recent_reconciliations = (
        ShiftReconciliation.query.order_by(ShiftReconciliation.reconciled_at.desc())
        .limit(20)
        .all()
    )

    # All users for the second signatory dropdown
    all_users = User.query.order_by(User.username).all()

    return render_template(
        "pharmacy/controlled_drugs.html",
        drug_summaries=drug_summaries,
        recent_dispenses=recent_dispenses,
        recent_reconciliations=recent_reconciliations,
        all_users=all_users,
        today=date.today().isoformat(),
    )


@bp.route("/controlled-drugs/api/ledger", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin", "doctor", "nursing", "medicine")
def controlled_drug_ledger():
    """
    GET /pharmacy/controlled-drugs/api/ledger?drug_id=<id>&limit=100
    Returns the full immutable ledger for a given controlled drug.
    """
    drug_id = request.args.get("drug_id", type=int)
    limit = request.args.get("limit", default=100, type=int)

    if not drug_id:
        return jsonify({"error": "drug_id query parameter is required"}), 400

    entries = (
        ControlledDrugBalance.query.filter_by(drug_id=drug_id)
        .order_by(ControlledDrugBalance.recorded_at.desc())
        .limit(limit)
        .all()
    )

    return jsonify(
        {
            "drug_id": drug_id,
            "count": len(entries),
            "ledger": [
                {
                    "id": e.id,
                    "transaction_type": e.transaction_type,
                    "quantity_change": e.quantity_change,
                    "balance_after": e.balance_after,
                    "recorded_at": e.recorded_at.strftime("%Y-%m-%d %H:%M"),
                    "notes": e.notes,
                }
                for e in entries
            ],
        }
    ), 200


@bp.route("/controlled-drugs/api/dispense", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def controlled_drug_dispense_api():
    """
    POST /pharmacy/controlled-drugs/api/dispense
    Dual-signature controlled drug dispense workflow.

    Required JSON:
      patient_id, drug_id, dose_mg, second_signatory_id, second_signatory_role
    """
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patient_id")
    drug_id = data.get("drug_id", type(None)) or data.get("drug_id")
    dose_mg = data.get("dose_mg")
    second_signatory_id = data.get("second_signatory_id")
    second_signatory_role = data.get("second_signatory_role", "pharmacist")

    if not all([patient_id, drug_id, dose_mg, second_signatory_id]):
        return jsonify(
            {
                "error": "patient_id, drug_id, dose_mg, and second_signatory_id are required"
            }
        ), 400

    try:
        dispense = dispense_controlled_drug(
            patient_id=str(patient_id),
            drug_id=int(drug_id),
            dose_mg=float(dose_mg),
            primary_pharmacist_id=current_user.id,
            second_signatory_id=int(second_signatory_id),
            second_signatory_role=str(second_signatory_role),
            user_id=current_user.id,
        )
        from extensions import db

        db.session.commit()

        return jsonify(
            {
                "status": "success",
                "message": f"Controlled drug dispensed and ledger updated. Balance after: {dispense.balance_after} mg",
                "dispense_id": dispense.id,
                "balance_after": dispense.balance_after,
            }
        ), 201

    except ControlledDrugError as exc:
        return jsonify({"error": str(exc)}), 422

    except Exception:
        logger.exception("Unexpected error during controlled drug dispense")
        return jsonify({"error": "Internal server error during dispense"}), 500


@bp.route("/controlled-drugs/api/reconcile", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def submit_reconciliation_api():
    """
    POST /pharmacy/controlled-drugs/api/reconcile
    Submit a shift-end physical count reconciliation.

    Required JSON:
      drug_id, shift_date (YYYY-MM-DD), shift_type (DAY|EVENING|NIGHT), physical_count
    """
    data = request.get_json(silent=True) or {}

    drug_id = data.get("drug_id")
    shift_date_str = data.get("shift_date")
    shift_type = (data.get("shift_type") or "DAY").strip().upper()
    physical_count = data.get("physical_count")

    if not all([drug_id, shift_date_str, physical_count is not None]):
        return jsonify(
            {
                "error": "drug_id, shift_date, shift_type, and physical_count are required"
            }
        ), 400

    try:
        shift_date = date.fromisoformat(shift_date_str)
    except ValueError:
        return jsonify({"error": "shift_date must be in YYYY-MM-DD format"}), 400

    try:
        recon = submit_shift_reconciliation(
            drug_id=int(drug_id),
            shift_date=shift_date,
            shift_type=shift_type,
            physical_count=float(physical_count),
            user_id=current_user.id,
        )
        from extensions import db

        db.session.commit()

        return jsonify(
            {
                "status": "success",
                "message": "Shift reconciliation completed — no discrepancy.",
                "reconciliation_id": recon.id,
                "variance": recon.variance,
                "status_flag": recon.status,
            }
        ), 201

    except ControlledDrugError as exc:
        from extensions import db

        db.session.rollback()
        return jsonify(
            {
                "error": str(exc),
                "status": "DISCREPANCY",
            }
        ), 422

    except Exception:
        logger.exception("Unexpected error during shift reconciliation")
        return jsonify({"error": "Internal server error during reconciliation"}), 500
