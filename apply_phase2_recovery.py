#!/usr/bin/env python3
"""Phase 2 recovery: fills in gaps from the aborted patch run."""
import os, sys, re

ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "departments/appointments/engine.py")):
    sys.exit("Run from the repository root.")


def has_text(path, needle):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return False
    return needle in open(p, encoding="utf-8").read()


def patch(path, old, new, count=1):
    p = os.path.join(ROOT, path)
    src = open(p, encoding="utf-8").read()
    # Normalize line endings
    src_norm = src.replace('\r\n', '\n').replace('\r', '\n')
    found = src_norm.count(old)
    if found == 0:
        print(f"  [skip] {path}: anchor not found (already patched or mismatched)")
        return False
    if found != count:
        sys.exit(f"FATAL: {path}: anchor found {found}x, expected {count}x. Nothing written.")
    new_src = src_norm.replace(old, new, count)
    open(p, "w", encoding="utf-8", newline='\n').write(new_src)
    print(f"patched  {path}")
    return True


def write_if_missing(path, content):
    p = os.path.join(ROOT, path)
    if os.path.exists(p):
        print(f"  [skip] {path}: already exists")
        return False
    open(p, "w", encoding="utf-8").write(content)
    print(f"created  {path}")
    return True


# ── 1. Fix migration script: add sys.path ───────────────────────────────────
migration_path = "scripts/migrate_encounter_stage.py"
if not has_text(migration_path, "sys.path.insert"):
    src = open(os.path.join(ROOT, migration_path), encoding="utf-8").read()
    src = 'import sys\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))\n\n' + src
    open(os.path.join(ROOT, migration_path), "w", encoding="utf-8").write(src)
    print(f"patched  {migration_path}")


# ── 2. Encounter: stage + transitions ───────────────────────────────────────
enc_path = "departments/models/encounter.py"
if not has_text(enc_path, "stage = db.Column"):
    # Read current file
    src = open(os.path.join(ROOT, enc_path), encoding="utf-8").read()
    
    # Check if logging already imported
    if "import logging" not in src:
        src = src.replace("import uuid", "import logging\nimport uuid", 1)
    
    # Check if logger already defined
    if 'logger = logging.getLogger(__name__)' not in src:
        src = src.replace("from extensions import db\n", 
                         "from extensions import db\n\nlogger = logging.getLogger(__name__)\n", 1)
    
    # Now inject the stage column and transitions before def close
    stage_code = '''
    # Visit lifecycle stage (Phase 2). Independent of `status`, which
    # billing/sync.py relies on to scope invoices to a visit.
    stage = db.Column(db.String(30), nullable=True, index=True)

    ALLOWED_STAGE_TRANSITIONS = {
        None: {"REGISTERED", "WAITING_DOCTOR", "IN_CONSULTATION"},
        "REGISTERED": {"WAITING_DOCTOR", "IN_CONSULTATION", "CANCELLED"},
        "WAITING_DOCTOR": {"IN_CONSULTATION", "CANCELLED"},
        "IN_CONSULTATION": {
            "IN_CONSULTATION",
            "AWAITING_RESULTS",
            "AWAITING_PHARMACY",
            "AWAITING_BILLING",
        },
        "AWAITING_RESULTS": {
            "IN_CONSULTATION",
            "AWAITING_RESULTS",
            "AWAITING_PHARMACY",
            "AWAITING_BILLING",
        },
        "AWAITING_PHARMACY": {"IN_CONSULTATION", "AWAITING_BILLING"},
        "AWAITING_BILLING": {"DISCHARGED"},
        "DISCHARGED": set(),
        "CANCELLED": set(),
    }

    def set_stage(self, new_stage: str) -> bool:
        """Advance the visit stage; refuses illegal transitions."""
        allowed = self.ALLOWED_STAGE_TRANSITIONS.get(
            self.stage, self.ALLOWED_STAGE_TRANSITIONS.get(None, set())
        )
        if new_stage not in allowed:
            logger.warning(
                "ENCOUNTER STAGE TRANSITION REFUSED: %s -> %s (encounter %s)",
                self.stage,
                new_stage,
                self.id,
            )
            return False
        self.stage = new_stage
        return True

'''
    # Insert before def close
    src = src.replace("    def close(self):", stage_code + "    def close(self):")
    
    # Update close() to set stage = DISCHARGED
    src = src.replace(
        '    def close(self):\n        """Marks the encounter as completed/discharged."""\n        self.status = "DISCHARGED"\n        self.ended_at = datetime.now(timezone.utc)\n',
        '    def close(self):\n        """Marks the encounter as completed/discharged."""\n        self.status = "DISCHARGED"\n        self.ended_at = datetime.now(timezone.utc)\n        self.stage = "DISCHARGED"\n'
    )
    
    open(os.path.join(ROOT, enc_path), "w", encoding="utf-8", newline='\n').write(src)
    print(f"patched  {enc_path}")
else:
    print(f"  [skip] {enc_path}: stage column already present")


# ── 3. QueueStatus: post-consult states ─────────────────────────────────────
if not has_text("departments/shared/queue_constants.py", "AWAITING_RESULTS"):
    patch("departments/shared/queue_constants.py",
          "    DISCHARGED = 7",
          "    DISCHARGED = 7\n"
          "    AWAITING_RESULTS = 8   # Consult done; patient at lab/imaging, may return\n"
          "    AWAITING_PHARMACY = 9  # Cleared to collect drugs\n"
          "    AWAITING_BILLING = 10  # Services done; awaiting settlement before exit")


# ── 4. Visit closure authority ──────────────────────────────────────────────
write_if_missing("departments/shared/visit_closure.py",
    '"""Phase 2: single authority for closing a visit (Encounter + legacy queue)."""\n'
    "import logging\n"
    "\n"
    "from departments.models.billing import Billing, DrugsBill, Invoice, InvoiceStatus\n"
    "from departments.models.encounter import Encounter\n"
    "from departments.models.medicine import PrescribedMedicine, RequestedImage, RequestedLab\n"
    "from departments.models.records import PatientWaitingList\n"
    "from departments.shared.queue_constants import QueueStatus\n"
    "from extensions import db\n"
    "\n"
    'logger = logging.getLogger(__name__)\n'
    "\n"
    "\n"
    "def has_pending_work(patient_id: str) -> bool:\n"
    '    if RequestedLab.query.filter_by(patient_id=patient_id, status=0).count():\n'
    "        return True\n"
    '    if RequestedImage.query.filter_by(patient_id=patient_id, status=0).count():\n'
    "        return True\n"
    '    if PrescribedMedicine.query.filter_by(patient_id=patient_id, status="0").count():\n'
    "        return True\n"
    "    if Billing.query.filter_by(patient_id=patient_id, status=0).count():\n"
    "        return True\n"
    "    if DrugsBill.query.filter_by(patient_id=patient_id, status=0).count():\n"
    "        return True\n"
    "    open_inv = Invoice.query.filter(\n"
    "        Invoice.patient_id == patient_id,\n"
    "        Invoice.status.in_(\n"
    "            [InvoiceStatus.DRAFT, InvoiceStatus.ISSUED, InvoiceStatus.PARTIAL]\n"
    "        ),\n"
    "        Invoice.balance > 0,\n"
    "    ).count()\n"
    "    return bool(open_inv)\n"
    "\n"
    "\n"
    "def active_encounter(patient_id: str):\n"
    "    return (\n"
    '        Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")\n'
    "        .order_by(Encounter.started_at.desc())\n"
    "        .first()\n"
    "    )\n"
    "\n"
    "\n"
    "def maybe_close_encounter(patient_id: str) -> bool:\n"
    '    """Close the visit only when all clinical work and all bills are settled."""\n'
    "    enc = active_encounter(patient_id)\n"
    "    if not enc or has_pending_work(patient_id):\n"
    "        return False\n"
    "    enc.close()\n"
    "    entry = PatientWaitingList.query.filter_by(patient_id=str(patient_id)).first()\n"
    "    if entry:\n"
    "        entry.seen = QueueStatus.DISCHARGED\n"
    "    db.session.commit()\n"
    '    logger.info("VISIT CLOSED: encounter %s for patient %s", enc.id, patient_id)\n'
    "    return True\n"
    "\n"
    "\n"
    "def force_close_visit(patient_id: str, reason: str = \"\") -> bool:\n"
    '    """Staff-initiated discharge regardless of pending work (AMA, transfer...)."""\n'
    "    enc = active_encounter(patient_id)\n"
    "    if not enc:\n"
    "        return False\n"
    "    enc.close()\n"
    "    entry = PatientWaitingList.query.filter_by(patient_id=str(patient_id)).first()\n"
    "    if entry:\n"
    "        entry.seen = QueueStatus.DISCHARGED\n"
    "    db.session.commit()\n"
    '    logger.info("VISIT FORCE-CLOSED (%s): encounter %s", reason or "manual", enc.id)\n'
    "    return True\n")


# ── 5. Engine: stage sync at triage ─────────────────────────────────────────
if not has_text("departments/appointments/engine.py", 'enc.set_stage("WAITING_DOCTOR")'):
    patch("departments/appointments/engine.py",
          "        appt.mark_ready()\n"
          "        db.session.commit()\n"
          '        logger.info("TRIAGE COMPLETE: Appointment %s is READY", appt.id)\n',
          "        appt.mark_ready()\n"
          "        enc = (\n"
          '            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")\n'
          "            .order_by(Encounter.started_at.desc())\n"
          "            .first()\n"
          "        )\n"
          "        if enc:\n"
          '            enc.set_stage("WAITING_DOCTOR")\n'
          "        db.session.commit()\n"
          '        logger.info("TRIAGE COMPLETE: Appointment %s is READY", appt.id)\n')


# ── 6. Consult: open → IN_CONSULTATION stage ────────────────────────────────
if not has_text("departments/medicine/consultations.py", 'open_enc.set_stage("IN_CONSULTATION")'):
    patch("departments/medicine/consultations.py",
          "        # Mark patient as IN_CONSULTATION\n"
          "        patient_entry.seen = QueueStatus.IN_CONSULTATION\n",
          "        # Mark patient as IN_CONSULTATION\n"
          "        patient_entry.seen = QueueStatus.IN_CONSULTATION\n"
          "        open_enc = (\n"
          '            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")\n'
          "            .order_by(Encounter.started_at.desc())\n"
          "            .first()\n"
          "        )\n"
          "        if open_enc:\n"
          '            open_enc.set_stage("IN_CONSULTATION")\n')


# ── 7. Consult: submit → route onward, never premature discharge ───────────
if not has_text("departments/medicine/consultations.py", "maybe_close_encounter"):
    patch("departments/medicine/consultations.py",
          "        # Update queue status to DISCHARGED and appointment to COMPLETED\n"
          "        waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()\n"
          "        if waiting_entry:\n"
          "            waiting_entry.seen = QueueStatus.DISCHARGED\n"
          "        appts = Appointment.query.filter(\n"
          "            Appointment.patient_id == str(patient_id),\n"
          '            Appointment.status.in_(["CHECKED_IN", "READY", "IN_PROGRESS"])\n'
          "        ).all()\n"
          "        for appt in appts:\n"
          '            appt.status = "COMPLETED"\n'
          "        db.session.commit()\n",
          "        # Phase 2: consult complete != visit complete. Route the visit to the\n"
          "        # next department and keep the Encounter open so post-consult charges\n"
          "        # (lab, drugs) still scope to this visit's invoice.\n"
          "        pending_labs = RequestedLab.query.filter_by(\n"
          "            patient_id=patient_id, status=0\n"
          "        ).count()\n"
          "        pending_imaging = RequestedImage.query.filter_by(\n"
          "            patient_id=patient_id, status=0\n"
          "        ).count()\n"
          "        pending_rx = PrescribedMedicine.query.filter_by(\n"
          '            patient_id=patient_id, status="0"\n'
          "        ).count()\n"
          "\n"
          "        waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()\n"
          "        encounter = (\n"
          '            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")\n'
          "            .order_by(Encounter.started_at.desc())\n"
          "            .first()\n"
          "        )\n"
          "        if pending_labs or pending_imaging:\n"
          '            next_seen, next_stage = QueueStatus.AWAITING_RESULTS, "AWAITING_RESULTS"\n'
          "        elif pending_rx:\n"
          '            next_seen, next_stage = QueueStatus.AWAITING_PHARMACY, "AWAITING_PHARMACY"\n'
          "        else:\n"
          '            next_seen, next_stage = QueueStatus.AWAITING_BILLING, "AWAITING_BILLING"\n'
          "        if waiting_entry:\n"
          "            waiting_entry.seen = next_seen\n"
          "        if encounter:\n"
          "            encounter.set_stage(next_stage)\n"
          "\n"
          "        appts = Appointment.query.filter(\n"
          "            Appointment.patient_id == str(patient_id),\n"
          '            Appointment.status.in_(["CHECKED_IN", "READY", "IN_PROGRESS"])\n'
          "        ).all()\n"
          "        for appt in appts:\n"
          '            appt.status = "COMPLETED"\n'
          "        db.session.commit()\n"
          "\n"
          "        # Close immediately only when nothing is pending and nothing is owed\n"
          "        # (e.g. a prepaid consult-only visit).\n"
          "        from departments.shared.visit_closure import maybe_close_encounter\n"
          "\n"
          "        maybe_close_encounter(str(patient_id))\n")


# ── 8. Recall route: stage sync ─────────────────────────────────────────────
if not has_text("departments/medicine/consultations.py", 'enc.set_stage("IN_CONSULTATION")'):
    patch("departments/medicine/consultations.py",
          "    if enc and enc.appointment_id:\n"
          "        appt = Appointment.query.get(enc.appointment_id)\n"
          '        if appt and appt.status in ("CHECKED_IN", "READY", "COMPLETED"):\n'
          '            appt.status = "IN_PROGRESS"\n'
          "    db.session.commit()\n",
          "    if enc:\n"
          '        enc.set_stage("IN_CONSULTATION")\n'
          "        if enc.appointment_id:\n"
          "            appt = Appointment.query.get(enc.appointment_id)\n"
          '            if appt and appt.status in ("CHECKED_IN", "READY", "COMPLETED"):\n'
          '                appt.status = "IN_PROGRESS"\n'
          "    db.session.commit()\n")


# ── 9. Explicit discharge route ─────────────────────────────────────────────
if not has_text("departments/medicine/consultations.py", "force_close_visit"):
    patch("departments/medicine/consultations.py",
          '@bp.route("/lab_patients")\n'
          "def lab_patients():\n",
          '@bp.route("/discharge/<patient_id>", methods=["POST"])\n'
          "@login_required\n"
          '@roles_required("medicine", "admin")\n'
          "def discharge_patient(patient_id):\n"
          '    """Staff-initiated discharge: closes the encounter even with pending work."""\n'
          "    from departments.shared.visit_closure import force_close_visit\n"
          "\n"
          '    if force_close_visit(str(patient_id), reason="manual discharge"):\n'
          '        flash(f"Patient {patient_id} discharged and encounter closed.", "success")\n'
          "    else:\n"
          '        flash("No active encounter found for this patient.", "error")\n'
          '    return redirect(url_for("medicine.index"))\n'
          "\n\n"
          '@bp.route("/lab_patients")\n'
          "def lab_patients():\n")


# ── 10. Closure hooks: invoice settlement + legacy pay_bills ────────────────
if not has_text("departments/models/billing.py", "maybe_close_encounter"):
    patch("departments/models/billing.py",
          "        if self.balance <= 0:\n"
          "            self.status = InvoiceStatus.PAID\n"
          "        elif self.amount_paid > 0:\n"
          "            self.status = InvoiceStatus.PARTIAL\n",
          "        if self.balance <= 0:\n"
          "            self.status = InvoiceStatus.PAID\n"
          "            # Fully settled: allow the visit to close if clinical work is done.\n"
          "            if self.encounter_id:\n"
          "                from departments.shared.visit_closure import maybe_close_encounter\n"
          "\n"
          "                maybe_close_encounter(self.patient_id)\n"
          "        elif self.amount_paid > 0:\n"
          "            self.status = InvoiceStatus.PARTIAL\n")


if not has_text("departments/billing/routes.py", "maybe_close_encounter"):
    patch("departments/billing/routes.py",
          '            DrugsBill.query.filter_by(patient_id=patient_id, status=0).update(\n'
          '                {"status": 1, "receipt_number": receipt_number}\n'
          "            )\n"
          "\n"
          "            db.session.commit()\n",
          '            DrugsBill.query.filter_by(patient_id=patient_id, status=0).update(\n'
          '                {"status": 1, "receipt_number": receipt_number}\n'
          "            )\n"
          "\n"
          "            db.session.commit()\n"
          "\n"
          "            # Phase 2: settlement may complete the visit.\n"
          "            from departments.shared.visit_closure import maybe_close_encounter\n"
          "\n"
          "            maybe_close_encounter(patient_id)\n")


print("\nRecovery complete. Run: python scripts/migrate_encounter_stage.py && pytest")