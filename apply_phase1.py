#!/usr/bin/env python3
"""Recovery script for the Phase-1 patches that aborted mid-run."""
import os

ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "departments/appointments/engine.py")):
    raise SystemExit("Run from repo root.")


def patch(path, old, new):
    p = os.path.join(ROOT, path)
    src = open(p, encoding="utf-8").read()
    if old not in src:
        print(f"  [skip] {path}: anchor already patched or missing")
        return
    open(p, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print(f"patched  {path}")


def write_if_missing(path, content):
    p = os.path.join(ROOT, path)
    if os.path.exists(p):
        print(f"  [skip] {path}: already exists")
        return
    open(p, "w", encoding="utf-8").write(content)
    print(f"created  {path}")


# ── 1. Fix soap_notes + submit_soap_notes to accept READY ────────────────────
# This is what broke test_patient_flow_integration.py.
patch(
    "departments/medicine/consultations.py",
    '            Appointment.status == "CHECKED_IN",\n',
    '            Appointment.status.in_(["CHECKED_IN", "READY"]),\n',
)
patch(
    "departments/medicine/consultations.py",
    '            Appointment.status.in_(["CHECKED_IN", "IN_PROGRESS"])\n',
    '            Appointment.status.in_(["CHECKED_IN", "READY", "IN_PROGRESS"])\n',
)

# ── 2. Insert Results Ready table into medicine/index.html ───────────────────
tpl = "departments/medicine/templates/medicine/index.html"
p = os.path.join(ROOT, tpl)
src = open(p, encoding="utf-8").read()
html = """
  <!-- Results Ready — recall to consult (Phase 1) -->
  <div class="md-table-card mt-4">
    <div class="md-table-header d-flex align-items-center justify-content-between">
      <h5 class="fw-bold mb-0 text-dark"><i class="bi bi-flask me-2" style="color: var(--md-primary);"></i>Results Ready — Recall to Consult</h5>
      <span class="badge px-3 py-2 rounded-pill" style="background: var(--md-primary);">{{ results_ready|length if results_ready else 0 }}</span>
    </div>
    <div class="table-responsive">
      <table class="table table-md table-hover align-middle mb-0">
        <thead>
          <tr><th>Patient ID</th><th>Name</th><th>Results completed</th><th class="text-end">Actions</th></tr>
        </thead>
        <tbody>
          {% for item in results_ready or [] %}
          <tr>
            <td><span class="badge bg-secondary font-monospace fs-6">{{ item.patient_id }}</span></td>
            <td class="fw-bold text-dark">{{ item.patient_name }}</td>
            <td><small class="text-muted">{{ item.completed_at or "—" }}</small></td>
            <td class="text-end">
              <form method="post" action="{{ url_for('medicine.recall_to_consult', patient_id=item.patient_id) }}" class="d-inline">
                <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                <button type="submit" class="btn-md-primary btn"><i class="bi bi-arrow-return-left me-1"></i>Recall to Consult</button>
              </form>
            </td>
          </tr>
          {% else %}
          <tr><td colspan="4" class="text-center py-4 text-muted">No completed results awaiting review.</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
"""
if "results_ready" not in src and "{% endblock %}" in src:
    idx = src.rfind("{% endblock %}")
    src = src[:idx] + html + src[idx:]
    open(p, "w", encoding="utf-8").write(src)
    print(f"patched  {tpl}")
else:
    print(f"  [skip] {tpl}: already patched or no endblock")

# ── 3. Create payment_gate.py ────────────────────────────────────────────────
write_if_missing(
    "departments/shared/payment_gate.py",
    """# Payment-state helper for the Phase-1 pharmacy dispensing gate.
from departments.models.billing import Billing, DrugsBill, Invoice, InvoiceStatus


def unpaid_charge_count(patient_id: str) -> int:
    count = Billing.query.filter_by(patient_id=patient_id, status=0).count()
    count += DrugsBill.query.filter_by(patient_id=patient_id, status=0).count()
    count += Invoice.query.filter(
        Invoice.patient_id == patient_id,
        Invoice.status.in_([InvoiceStatus.DRAFT, InvoiceStatus.ISSUED]),
        Invoice.balance > 0,
    ).count()
    return count


def has_unpaid_charges(patient_id: str) -> bool:
    return unpaid_charge_count(patient_id) > 0
""",
)

# ── 4. Patch pharmacy/dispensing.py ──────────────────────────────────────────
dp = "departments/pharmacy/dispensing.py"
src = open(os.path.join(ROOT, dp), encoding="utf-8").read()

if "current_app" not in src:
    src = src.replace(
        "from flask import flash, redirect, render_template, request, url_for\nfrom flask_login import login_required",
        "from flask import current_app, flash, redirect, render_template, request, url_for\nfrom flask_login import current_user, login_required",
    )

if "has_unpaid_charges" not in src:
    src = src.replace(
        "from departments.models.records import Patient\nfrom departments.rbac import roles_required\nfrom extensions import db",
        "from departments.models.admin import Log\nfrom departments.models.records import Patient\nfrom departments.rbac import roles_required\nfrom departments.shared.payment_gate import has_unpaid_charges\nfrom extensions import db",
    )

if "has_unpaid_charges(prescribed_medicines" not in src:
    src = src.replace(
        "        drugs = Drug.query.all()\n        drug_batches = {",
        '        if has_unpaid_charges(prescribed_medicines[0].patient_id):\n'
        "            flash(\n"
        '                "Warning: this patient has unsettled charges — dispensing on credit."\n'
        '                " Set PHARMACY_REQUIRE_PAID=True to enforce payment first.",\n'
        '                "warning",\n'
        "            )\n\n"
        "        drugs = Drug.query.all()\n"
        "        drug_batches = {",
    )

if "Phase-1 payment gate" not in src:
    src = src.replace(
        "        patient_id = prescribed_medicines[0].patient_id\n\n        # Process each drug in the form\n",
        "        patient_id = prescribed_medicines[0].patient_id\n\n"
        "        # Phase-1 payment gate\n"
        "        unpaid = has_unpaid_charges(patient_id)\n"
        '        if unpaid and current_app.config.get("PHARMACY_REQUIRE_PAID", False):\n'
        "            flash(\n"
        '                "Dispensing blocked: patient has unsettled charges. "\n'
        '                "Complete billing first (or disable PHARMACY_REQUIRE_PAID).",\n'
        '                "error",\n'
        "            )\n"
        '            return redirect(url_for("pharmacy.view_prescriptions", patient_id=patient_id))\n'
        "        if unpaid:\n"
        "            db.session.add(\n"
        "                Log(\n"
        '                    level="WARNING",\n'
        '                    message=f"Pharmacy dispensing on credit for patient {patient_id} "\n'
        '                    f"(prescription {prescription_id}) with unsettled charges.",\n'
        "                    user_id=current_user.id,\n"
        '                    source="pharmacy",\n'
        "                )\n"
        "            )\n\n"
        "        # Process each drug in the form\n",
    )

open(os.path.join(ROOT, dp), "w", encoding="utf-8").write(src)
print(f"patched  {dp}")

# ── 5. Create test file ──────────────────────────────────────────────────────
write_if_missing(
    "tests/test_phase1_queue_bridge.py",
    """# Phase-1 queue bridge: READY-state transitions across the merged queues.
from departments.appointments.engine import ScheduleEngine
from departments.appointments.models import Appointment


def test_walk_in_starts_checked_in(app):
    appt = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    assert appt.status == "CHECKED_IN"


def test_triage_complete_moves_to_ready(app):
    appt = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    ready = ScheduleEngine().mark_triage_complete("P0001")
    assert ready is not None and ready.id == appt.id
    assert ready.status == "READY"


def test_mark_triage_complete_is_idempotent_noop(app):
    assert ScheduleEngine().mark_triage_complete("P9999") is None
    appt = ScheduleEngine().create_walk_in(patient_id="P0002", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0002")
    assert ScheduleEngine().mark_triage_complete("P0002") is None
    assert Appointment.query.get(appt.id).status == "READY"


def test_call_in_accepts_ready(app):
    appt = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0001")
    called = ScheduleEngine().call_in(appt.id)
    assert called is not None and called.status == "IN_PROGRESS"


def test_live_queue_includes_ready_excludes_in_progress(app):
    a1 = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    a2 = ScheduleEngine().create_walk_in(patient_id="P0002", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0002")
    ScheduleEngine().call_in(a1.id)
    ids = {a.id for a in ScheduleEngine().get_live_queue()}
    assert a2.id in ids and a1.id not in ids
""",
)

print("\nRecovery complete. Run: pytest -v")