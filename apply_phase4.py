#!/usr/bin/env python3
"""Phase 4: Retire PatientWaitingList writes. Encounter is now the single source of truth."""
import os, sys

ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "departments/appointments/engine.py")):
    sys.exit("Run from the repository root.")


def patch(path, old, new, count=1):
    p = os.path.join(ROOT, path)
    src = open(p, encoding="utf-8").read()
    found = src.count(old)
    if found == 0:
        print(f"  [skip] {path}: anchor not found (already patched?)")
        return False
    if found != count:
        sys.exit(f"FATAL: {path}: anchor found {found}x, expected {count}x. Nothing written.")
    open(p, "w", encoding="utf-8").write(src.replace(old, new, count))
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


# ── 1. Encounter: add backward-compat properties for templates ──────────────
patch(
    "departments/models/encounter.py",
    '    patient = db.relationship("Patient", foreign_keys="Encounter.patient_id",\n'
    '                              primaryjoin="Encounter.patient_id == Patient.patient_id",\n'
    '                              lazy="joined", viewonly=True)\n',
    '    patient = db.relationship("Patient", foreign_keys="Encounter.patient_id",\n'
    '                              primaryjoin="Encounter.patient_id == Patient.patient_id",\n'
    '                              lazy="joined", viewonly=True)\n'
    '\n'
    '    @property\n'
    '    def seen(self):\n'
    '        """Backward compat: maps stage to legacy QueueStatus integer for templates."""\n'
    '        from departments.shared.queue_constants import QueueStatus\n'
    '        stage_map = {\n'
    '            "REGISTERED": QueueStatus.WAITING_TRIAGE,\n'
    '            "WAITING_DOCTOR": QueueStatus.VITALS_DONE,\n'
    '            "IN_CONSULTATION": QueueStatus.IN_CONSULTATION,\n'
    '            "AWAITING_RESULTS": QueueStatus.AWAITING_RESULTS,\n'
    '            "AWAITING_PHARMACY": QueueStatus.AWAITING_PHARMACY,\n'
    '            "AWAITING_BILLING": QueueStatus.AWAITING_BILLING,\n'
    '            "DISCHARGED": QueueStatus.DISCHARGED,\n'
    '        }\n'
    '        return stage_map.get(self.stage, 0)\n'
    '\n'
    '    @property\n'
    '    def last_updated(self):\n'
    '        """Backward compat for templates expecting .last_updated."""\n'
    '        return self.started_at\n',
)


# ── 2. records/routes.py: remove PatientWaitingList writes ──────────────────
patch(
    "departments/records/routes.py",
    "        waiting_entry = PatientWaitingList(patient_id=new_p.patient_id, seen=QueueStatus.WAITING_TRIAGE)\n"
    "        db.session.add(waiting_entry)\n"
    "        db.session.commit()\n"
    "\n"
    "        # Bridge registration to live queue via Appointment\n",
    "        # Bridge registration to live queue via Appointment\n",
)

patch(
    "departments/records/routes.py",
    "    waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()\n"
    "    if not waiting_entry:\n"
    "        waiting_entry = PatientWaitingList(patient_id=patient_id, seen=QueueStatus.WAITING_TRIAGE)\n"
    "        db.session.add(waiting_entry)\n"
    "    else:\n"
    "        waiting_entry.seen = QueueStatus.WAITING_TRIAGE\n"
    "\n"
    "    db.session.commit()\n"
    "\n"
    "    provider_id = str(getattr(current_user, \"id\", \"1\") or \"1\")\n",
    "    db.session.commit()\n"
    "\n"
    "    provider_id = str(getattr(current_user, \"id\", \"1\") or \"1\")\n",
)

patch(
    "departments/records/routes.py",
    "@bp.route(\"/waiting_list\")\n"
    "@login_required\n"
    "@roles_required(\"records\", \"admin\")\n"
    "def waiting_list():\n"
    "    waiting_list = (\n"
    "        db.session.query(PatientWaitingList, Patient)\n"
    "        .join(Patient, PatientWaitingList.patient_id == Patient.patient_id)\n"
    "        .all()\n"
    "    )\n"
    "    return render_template(\"records/waiting_list.html\", waiting_list=waiting_list)\n",
    "@bp.route(\"/waiting_list\")\n"
    "@login_required\n"
    "@roles_required(\"records\", \"admin\")\n"
    "def waiting_list():\n"
    "    # Phase 4: read from Encounter instead of PatientWaitingList\n"
    "    from departments.models.encounter import Encounter\n"
    "\n"
    "    waiting_list = (\n"
    "        db.session.query(Encounter, Patient)\n"
    "        .join(Patient, Encounter.patient_id == Patient.patient_id)\n"
    '        .filter(Encounter.status == "ACTIVE")\n'
    "        .order_by(Encounter.started_at.asc())\n"
    "        .all()\n"
    "    )\n"
    "    return render_template(\"records/waiting_list.html\", waiting_list=waiting_list)\n",
)


# ── 3. nursing/vitals.py: remove PatientWaitingList writes ──────────────────
patch(
    "departments/nursing/vitals.py",
    "            db.session.add(vitals_data)\n"
    "            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()\n"
    "            if waiting_entry:\n"
    "                waiting_entry.seen = QueueStatus.VITALS_DONE\n"
    "            db.session.commit()\n",
    "            db.session.add(vitals_data)\n"
    "            db.session.commit()\n",
)

patch(
    "departments/nursing/vitals.py",
    "            db.session.add(new_vital_sign)\n"
    "            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()\n"
    "            if waiting_entry:\n"
    "                waiting_entry.seen = QueueStatus.VITALS_DONE\n"
    "            db.session.commit()\n",
    "            db.session.add(new_vital_sign)\n"
    "            db.session.commit()\n",
)


# ── 4. medicine/consultations.py: remove PatientWaitingList writes ──────────
patch(
    "departments/medicine/consultations.py",
    "        # Fetch the patient from the waiting list\n"
    "        patient_entry = (\n"
    "            PatientWaitingList.query.filter_by(patient_id=patient_id)\n"
    "            .options(joinedload(PatientWaitingList.patient))\n"
    "            .first()\n"
    "        )\n"
    "        if not patient_entry or not patient_entry.patient:\n"
    "            flash(\n"
    "                f\"Patient with ID {patient_id} not found in the waiting list!\", \"error\"\n"
    "            )\n"
    "            return redirect(\n"
    "                url_for(\"medicine.index\")\n"
    "            )  # Redirect to index if patient not found\n"
    "        patient = patient_entry.patient\n"
    "\n"
    "        # Mark patient as IN_CONSULTATION\n"
    "        patient_entry.seen = QueueStatus.IN_CONSULTATION\n"
    "        open_enc = (\n"
    '            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")\n'
    "            .order_by(Encounter.started_at.desc())\n"
    "            .first()\n"
    "        )\n"
    "        if open_enc:\n"
    '            open_enc.set_stage("IN_CONSULTATION")\n',
    "        # Fetch the patient\n"
    "        patient = Patient.query.filter_by(patient_id=patient_id).first()\n"
    "        if not patient:\n"
    "            flash(f\"Patient with ID {patient_id} not found!\", \"error\")\n"
    "            return redirect(url_for(\"medicine.index\"))\n"
    "\n"
    "        # Mark patient as IN_CONSULTATION\n"
    "        open_enc = (\n"
    '            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")\n'
    "            .order_by(Encounter.started_at.desc())\n"
    "            .first()\n"
    "        )\n"
    "        if open_enc:\n"
    '            open_enc.set_stage("IN_CONSULTATION")\n',
)

patch(
    "departments/medicine/consultations.py",
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
    "            encounter.set_stage(next_stage)\n",
    "        encounter = (\n"
    '            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")\n'
    "            .order_by(Encounter.started_at.desc())\n"
    "            .first()\n"
    "        )\n"
    "        if pending_labs or pending_imaging:\n"
    '            next_stage = "AWAITING_RESULTS"\n'
    "        elif pending_rx:\n"
    '            next_stage = "AWAITING_PHARMACY"\n'
    "        else:\n"
    '            next_stage = "AWAITING_BILLING"\n'
    "        if encounter:\n"
    "            encounter.set_stage(next_stage)\n",
)


# ── 5. shared/visit_closure.py: remove PatientWaitingList writes ────────────
patch(
    "departments/shared/visit_closure.py",
    "    enc.close()\n"
    "    entry = PatientWaitingList.query.filter_by(patient_id=str(patient_id)).first()\n"
    "    if entry:\n"
    "        entry.seen = QueueStatus.DISCHARGED\n"
    "    db.session.commit()\n",
    "    enc.close()\n"
    "    db.session.commit()\n",
    count=2,
)


# ── 6. Tests ─────────────────────────────────────────────────────────────────
write_if_missing(
    "tests/test_phase4_retire_waiting_list.py",
    "# Phase 4: PatientWaitingList writes are retired. Encounter is the source of truth.\n"
    "from datetime import date\n"
    "\n"
    "from departments.appointments.engine import ScheduleEngine\n"
    "from departments.models.encounter import Encounter\n"
    "from departments.models.records import Patient, PatientWaitingList\n"
    "from departments.shared.queue_constants import QueueStatus\n"
    "from extensions import db\n"
    "\n"
    "\n"
    "def _patient(pid):\n"
    "    p = Patient(\n"
    "        patient_id=pid, name=f\"Test {pid}\", sex=\"F\",\n"
    "        date_of_birth=date(1990, 1, 1),\n"
    "    )\n"
    "    db.session.add(p)\n"
    "    db.session.commit()\n"
    "    return p\n"
    "\n"
    "\n"
    "def test_registration_does_not_create_waiting_list_row(app):\n"
    '    _patient("P0001")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    "    # Phase 4: no PatientWaitingList row should be created\n"
    "    assert PatientWaitingList.query.filter_by(patient_id=\"P0001\").first() is None\n"
    "    # But Encounter should exist\n"
    "    enc = Encounter.query.filter_by(patient_id=\"P0001\").first()\n"
    "    assert enc is not None\n"
    '    assert enc.stage == "REGISTERED"\n'
    "\n"
    "\n"
    "def test_encounter_has_seen_property_for_backward_compat(app):\n"
    '    _patient("P0001")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    "    enc = Encounter.query.filter_by(patient_id=\"P0001\").first()\n"
    "    # The .seen property should map stage to legacy QueueStatus integer\n"
    "    assert enc.seen == QueueStatus.WAITING_TRIAGE\n"
    '    ScheduleEngine().mark_triage_complete("P0001")\n'
    "    assert enc.seen == QueueStatus.VITALS_DONE\n"
    "\n"
    "\n"
    "def test_encounter_has_last_updated_property(app):\n"
    '    _patient("P0001")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    "    enc = Encounter.query.filter_by(patient_id=\"P0001\").first()\n"
    "    # The .last_updated property should return started_at\n"
    "    assert enc.last_updated == enc.started_at\n",
)


print("\nPhase 4 applied. Run: pytest tests/test_phase4_retire_waiting_list.py -v && pytest")