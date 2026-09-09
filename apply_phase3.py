#!/usr/bin/env python3
"""Phase 3: QueueService — department queues read from Encounter.stage."""
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


# ── 1. Encounter: add patient relationship ──────────────────────────────────
# So templates iterating Encounters can still do `entry.patient.name` etc.
patch(
    "departments/models/encounter.py",
    'from extensions import db\n\nlogger = logging.getLogger(__name__)\n',
    'from extensions import db\n\nlogger = logging.getLogger(__name__)\n'
    "\n"
    "# Late import used only in the relationship; declared here so templates\n"
    "# iterating over Encounters can access `encounter.patient` directly.\n"
    'from departments.models.records import Patient  # noqa: E402\n',
)

patch(
    "departments/models/encounter.py",
    '    started_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))\n'
    "    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)\n",
    '    started_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))\n'
    "    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)\n"
    "\n"
    '    patient = db.relationship("Patient", foreign_keys="Encounter.patient_id",\n'
    '                              primaryjoin="Encounter.patient_id == Patient.patient_id",\n'
    '                              lazy="joined", viewonly=True)\n',
)


# ── 2. QueueService ─────────────────────────────────────────────────────────
write_if_missing(
    "departments/shared/queue_service.py",
    '"""Phase 3: department queue reader built on Encounter.stage.\n'
    "\n"
    "Replaces direct PatientWaitingList.seen queries in department index routes.\n"
    "PatientWaitingList writes continue (dual-write) for one release as a\n"
    "rollback safety net; Phase 4 removes them.\n"
    '"""\n'
    "from departments.models.encounter import Encounter\n"
    "\n"
    "\n"
    "# Which Encounter.stage values count as 'queued' for each department.\n"
    "DEPARTMENT_STAGES = {\n"
    '    "nursing": ["REGISTERED"],  # waiting for triage/vitals\n'
    '    "medicine": ["WAITING_DOCTOR", "IN_CONSULTATION", "AWAITING_RESULTS"],\n'
    '    "laboratory": ["AWAITING_RESULTS"],\n'
    '    "pharmacy": ["AWAITING_PHARMACY"],\n'
    '    "billing": ["AWAITING_BILLING"],\n'
    "}\n"
    "\n"
    "\n"
    "def queue_for(department: str):\n"
    '    """Return ACTIVE encounters currently queued for the given department.\n'
    "\n"
    "    Returns a list of Encounter objects (each with a `.patient` joined\n"
    "    relationship) so templates iterating over the list can use the same\n"
    "    `entry.patient.name` accessors they used on PatientWaitingList rows.\n"
    '    """\n'
    "    stages = DEPARTMENT_STAGES.get(department)\n"
    "    if not stages:\n"
    "        return []\n"
    "    return (\n"
    '        Encounter.query.filter(\n'
    '            Encounter.status == "ACTIVE",\n'
    "            Encounter.stage.in_(stages),\n"
    "        )\n"
    "        .order_by(Encounter.started_at.asc())\n"
    "        .all()\n"
    "    )\n"
    "\n"
    "\n"
    "def count_for(department: str) -> int:\n"
    "    stages = DEPARTMENT_STAGES.get(department)\n"
    "    if not stages:\n"
    "        return 0\n"
    '    return Encounter.query.filter(\n'
    '        Encounter.status == "ACTIVE",\n'
    "        Encounter.stage.in_(stages),\n"
    "    ).count()\n",
)


# ── 3. Nursing index: read from QueueService ────────────────────────────────
patch(
    "departments/nursing/vitals.py",
    "from departments.models.records import Patient, PatientWaitingList\n",
    "from departments.models.records import Patient, PatientWaitingList\n"
    "from departments.shared import queue_service\n",
)

patch(
    "departments/nursing/vitals.py",
    '    """Display the nursing waiting list."""\n'
    "    try:\n"
    "        # Fetch all patients in the nursing waiting list who are not yet seen\n"
    "        nursing_waiting_list = (\n"
    "            PatientWaitingList.query.filter_by(seen=4)\n"
    "            .options(\n"
    "                joinedload(PatientWaitingList.patient)  # Eager load patient details\n"
    "            )\n"
    "            .all()\n"
    "        )\n"
    "\n"
    "        # Filter out entries with missing patient relationships\n"
    "        valid_waiting_list = [\n"
    "            entry\n"
    "            for entry in nursing_waiting_list\n"
    "            if entry.patient  # Ensure patient relationship exists\n"
    "        ]\n"
    "\n"
    '        return render_template("nursing/index.html", waiting_list=valid_waiting_list)\n',
    '    """Display the nursing waiting list."""\n'
    "    try:\n"
    "        # Phase 3: read from Encounter.stage via QueueService.\n"
    "        valid_waiting_list = queue_service.queue_for(\"nursing\")\n"
    "\n"
    '        return render_template("nursing/index.html", waiting_list=valid_waiting_list)\n',
)


# ── 4. Medicine index: read from QueueService ───────────────────────────────
patch(
    "departments/medicine/consultations.py",
    "from departments.shared.queue_constants import QueueStatus\n"
    "from extensions import db\n",
    "from departments.shared import queue_service\n"
    "from departments.shared.queue_constants import QueueStatus\n"
    "from extensions import db\n",
)

patch(
    "departments/medicine/consultations.py",
    '    """Display the medicine waiting list."""\n'
    "    try:\n"
    "        # Fetch all patients in the medicine waiting list (waiting triage, vitals done, or in consultation)\n"
    "        waiting_list = (\n"
    "            PatientWaitingList.query.filter(\n"
    "                PatientWaitingList.seen.in_(\n"
    "                    [QueueStatus.WAITING_TRIAGE, QueueStatus.VITALS_DONE, QueueStatus.IN_CONSULTATION]\n"
    "                )\n"
    "            )\n"
    "            .options(joinedload(PatientWaitingList.patient))\n"
    "            .all()\n"
    "        )\n"
    "        # Filter out invalid entries (e.g., missing patient relationships)\n"
    "        valid_waiting_list = [entry for entry in waiting_list if entry.patient]\n",
    '    """Display the medicine waiting list."""\n'
    "    try:\n"
    "        # Phase 3: read from Encounter.stage via QueueService.\n"
    "        valid_waiting_list = queue_service.queue_for(\"medicine\")\n",
)


# ── 5. Tests ─────────────────────────────────────────────────────────────────
write_if_missing(
    "tests/test_phase3_queue_service.py",
    "# Phase 3: QueueService reads department queues from Encounter.stage.\n"
    "from datetime import date\n"
    "\n"
    "from departments.appointments.engine import ScheduleEngine\n"
    "from departments.models.encounter import Encounter\n"
    "from departments.models.records import Patient, PatientWaitingList\n"
    "from departments.shared import queue_service\n"
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
    "    db.session.add(PatientWaitingList(patient_id=pid, seen=QueueStatus.WAITING_TRIAGE))\n"
    "    db.session.commit()\n"
    "    return p\n"
    "\n"
    "\n"
    "def test_nursing_queue_shows_registered_only(app):\n"
    '    _patient("P0001")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    "    out = queue_service.queue_for(\"nursing\")\n"
    "    assert len(out) == 1\n"
    '    assert out[0].stage == "REGISTERED"\n'
    "    assert out[0].patient.name == \"Test P0001\"\n"
    "\n"
    "\n"
    "def test_nursing_queue_excludes_patients_past_triage(app):\n"
    '    _patient("P0001")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    '    ScheduleEngine().mark_triage_complete("P0001")\n'
    "    assert queue_service.queue_for(\"nursing\") == []\n"
    "\n"
    "\n"
    "def test_medicine_queue_includes_waiting_and_in_consult(app):\n"
    '    _patient("P0001")\n'
    '    _patient("P0002")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    "    ScheduleEngine().create_walk_in(patient_id=\"P0002\")\n"
    '    ScheduleEngine().mark_triage_complete("P0001")\n'
    "    enc1 = Encounter.query.filter_by(patient_id=\"P0001\").first()\n"
    '    enc1.set_stage("IN_CONSULTATION")\n'
    "    db.session.commit()\n"
    "    out = queue_service.queue_for(\"medicine\")\n"
    "    ids = {e.patient_id for e in out}\n"
    "    assert \"P0001\" in ids and \"P0002\" in ids\n"
    "\n"
    "\n"
    "def test_medicine_queue_excludes_discharged(app):\n"
    '    _patient("P0001")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    "    enc = Encounter.query.filter_by(patient_id=\"P0001\").first()\n"
    "    enc.close()\n"
    "    db.session.commit()\n"
    "    assert queue_service.queue_for(\"medicine\") == []\n"
    "\n"
    "\n"
    "def test_count_for_returns_correct_totals(app):\n"
    '    _patient("P0001")\n'
    '    _patient("P0002")\n'
    "    ScheduleEngine().create_walk_in(patient_id=\"P0001\")\n"
    "    ScheduleEngine().create_walk_in(patient_id=\"P0002\")\n"
    '    assert queue_service.count_for("nursing") == 2\n'
    '    ScheduleEngine().mark_triage_complete("P0001")\n'
    '    assert queue_service.count_for("nursing") == 1\n'
    '    assert queue_service.count_for("medicine") == 1\n'
    "\n"
    "\n"
    "def test_unknown_department_returns_empty(app):\n"
    '    assert queue_service.queue_for("radiology") == []\n'
    '    assert queue_service.count_for("radiology") == 0\n',
)


print("\nPhase 3 applied. Run: pytest tests/test_phase3_queue_service.py -v && pytest")