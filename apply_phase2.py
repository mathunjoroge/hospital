#!/usr/bin/env python3
"""Phase 2: Encounter stage machine + single visit-closure authority."""
import os, sys

ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "departments/appointments/engine.py")):
    sys.exit("Run from the repository root.")


def patch(path, old, new, count=1):
    p = os.path.join(ROOT, path)
    src = open(p, encoding="utf-8").read()
    found = src.count(old)
    if found != count:
        sys.exit(f"FATAL: {path}: anchor found {found}x, expected {count}x. Nothing written.")
    open(p, "w", encoding="utf-8").write(src.replace(old, new, count))
    print("patched  ", path)


def write_if_missing(path, content):
    p = os.path.join(ROOT, path)
    if os.path.exists(p):
        sys.exit(f"FATAL: {path} already exists. Aborting.")
    open(p, "w", encoding="utf-8").write(content)
    print("created  ", path)


# ── 1. Encounter: stage column + guarded transitions ────────────────────────
patch("departments/models/encounter.py",
      "import uuid\n",
      "import logging\nimport uuid\n")

patch("departments/models/encounter.py",
      "from extensions import db\n",
      "from extensions import db\n\nlogger = logging.getLogger(__name__)\n")

patch("departments/models/encounter.py",
      "    # ACTIVE, DISCHARGED, CANCELLED, ABORTED\n"
      '    status = db.Column(db.String(20), nullable=False, default="ACTIVE")\n'
      "\n"
      "    def close(self):\n"
      '        """Marks the encounter as completed/discharged."""\n'
      '        self.status = "DISCHARGED"\n'
      "        self.ended_at = datetime.now(timezone.utc)\n",
      "    # ACTIVE, DISCHARGED, CANCELLED, ABORTED\n"
      '    status = db.Column(db.String(20), nullable=False, default="ACTIVE")\n'
      "\n"
      "    # Visit lifecycle stage (Phase 2). Independent of `status`, which\n"
      "    # billing/sync.py relies on to scope invoices to a visit.\n"
      "    stage = db.Column(db.String(30), nullable=True, index=True)\n"
      "\n"
      "    ALLOWED_STAGE_TRANSITIONS = {\n"
      '        None: {"REGISTERED", "WAITING_DOCTOR", "IN_CONSULTATION"},\n'
      '        "REGISTERED": {"WAITING_DOCTOR", "IN_CONSULTATION", "CANCELLED"},\n'
      '        "WAITING_DOCTOR": {"IN_CONSULTATION", "CANCELLED"},\n'
      '        "IN_CONSULTATION": {\n'
      '            "IN_CONSULTATION",\n'
      '            "AWAITING_RESULTS",\n'
      '            "AWAITING_PHARMACY",\n'
      '            "AWAITING_BILLING",\n'
      "        },\n"
      '        "AWAITING_RESULTS": {\n'
      '            "IN_CONSULTATION",\n'
      '            "AWAITING_RESULTS",\n'
      '            "AWAITING_PHARMACY",\n'
      '            "AWAITING_BILLING",\n'
      "        },\n"
      '        "AWAITING_PHARMACY": {"IN_CONSULTATION", "AWAITING_BILLING"},\n'
      '        "AWAITING_BILLING": {"DISCHARGED"},\n'
      '        "DISCHARGED": set(),\n'
      '        "CANCELLED": set(),\n'
      "    }\n"
      "\n"
      "    def set_stage(self, new_stage: str) -> bool:\n"
      '        """Advance the visit stage; refuses illegal transitions."""\n'
      "        allowed = self.ALLOWED_STAGE_TRANSITIONS.get(\n"
      "            self.stage, self.ALLOWED_STAGE_TRANSITIONS.get(None, set())\n"
      "        )\n"
      "        if new_stage not in allowed:\n"
      "            logger.warning(\n"
      '                "ENCOUNTER STAGE TRANSITION REFUSED: %s -> %s (encounter %s)",\n'
      "                self.stage,\n"
      "                new_stage,\n"
      "                self.id,\n"
      "            )\n"
      "            return False\n"
      "        self.stage = new_stage\n"
      "        return True\n"
      "\n"
      "    def close(self):\n"
      '        """Marks the encounter as completed/discharged."""\n'
      '        self.status = "DISCHARGED"\n'
      "        self.ended_at = datetime.now(timezone.utc)\n"
      '        self.stage = "DISCHARGED"\n')

# ── 2. QueueStatus: post-consult states ─────────────────────────────────────
patch("departments/shared/queue_constants.py",
      "    DISCHARGED = 7",
      "    DISCHARGED = 7\n"
      "    AWAITING_RESULTS = 8   # Consult done; patient at lab/imaging, may return\n"
      "    AWAITING_PHARMACY = 9  # Cleared to collect drugs\n"
      "    AWAITING_BILLING = 10  # Services done; awaiting settlement before exit")

# ── 3. Single closure authority ─────────────────────────────────────────────
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
    "def force_close_visit(patient_id: str, reason: str = "") -> bool:\n"
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

# ── 4. Engine: stage sync at triage ─────────────────────────────────────────
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

# ── 5. Consult: open → IN_CONSULTATION stage ────────────────────────────────
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

# ── 6. Consult: submit → route onward, never premature discharge ───────────
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

# ── 7. Recall route: stage sync ─────────────────────────────────────────────
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

# ── 8. Explicit discharge route ─────────────────────────────────────────────
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

# ── 9. Closure hooks: invoice settlement + legacy pay_bills ─────────────────
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

print("\nPhase 2 applied. Run: python scripts/migrate_encounter_stage.py && pytest")