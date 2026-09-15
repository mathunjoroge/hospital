"""
tests/invariants/test_clinical_auth_invariants.py
──────────────────────────────────────────────────
Invariant: every clinical-write endpoint MUST reject unauthenticated requests.

These tests encode the three P0 gaps that were found and fixed:
  1. /nursing/mar/chart      — MAR medication charting
  2. /nursing/mar/auto_bill  — daily billing trigger
  3. /pharmacy/fefo/dispense — FEFO stock dispensing

A regression means an unauthenticated caller can write clinical or financial
records to the database, which is an immediate patient-safety and integrity
failure.  These tests must NEVER be removed or weakened.
"""



# ─── helpers ────────────────────────────────────────────────────────────────

def _assert_auth_required(client, method: str, url: str, payload=None):
    """Assert that a route returns 401 or 302 (login redirect) when not logged in."""
    fn = getattr(client, method)
    if payload is not None:
        resp = fn(url, json=payload)
    else:
        resp = fn(url)

    assert resp.status_code in (401, 302, 403), (
        f"INVARIANT VIOLATION: {method.upper()} {url} returned {resp.status_code} "
        f"without authentication. This route MUST require login."
    )


# ─── MAR charting ────────────────────────────────────────────────────────────

class TestMARAuthRequired:
    """
    The Medication Administration Record endpoints record which nurse gave
    which drug to which patient. They MUST be gated by authentication and
    a nursing role check.
    """

    def test_chart_medication_rejects_anonymous(self, client):
        """Unauthenticated POST to /nursing/mar/chart must be rejected."""
        _assert_auth_required(
            client,
            "post",
            "/nursing/mar/chart",
            payload={
                "patient_id": "P-0001",
                "medication": "Morphine 10mg",
                "dosage": "10mg IV",
                # nurse_id deliberately omitted — if the route accepted this
                # it would also mean identity is not verified
            },
        )

    def test_auto_bill_rejects_anonymous(self, client):
        """Unauthenticated POST to /nursing/mar/auto_bill must be rejected.

        If this fails, any HTTP client can trigger billing charges for every
        admitted patient without any credentials.
        """
        _assert_auth_required(client, "post", "/nursing/mar/auto_bill")

    def test_ward_occupancy_rejects_anonymous(self, client):
        """Unauthenticated GET to /nursing/mar/occupancy must be rejected."""
        _assert_auth_required(client, "get", "/nursing/mar/occupancy")


# ─── FEFO pharmacy ───────────────────────────────────────────────────────────

class TestFEFOAuthRequired:
    """
    FEFO endpoints perform actual stock deductions and create DispensedDrug
    records.  They must always require a pharmacist session.
    """

    def test_dispense_rejects_anonymous(self, client):
        """Unauthenticated POST to /pharmacy/fefo/dispense must be rejected.

        If this fails, any HTTP client can remove stock for any drug for any
        patient with no audit trail linking to a real pharmacist.
        """
        _assert_auth_required(
            client,
            "post",
            "/pharmacy/fefo/dispense",
            payload={"patient_id": "P-0001", "drug_id": 1, "quantity": 10},
        )

    def test_allocate_preview_rejects_anonymous(self, client):
        """Unauthenticated GET to /pharmacy/fefo/allocate must be rejected."""
        _assert_auth_required(client, "get", "/pharmacy/fefo/allocate?drug_id=1&quantity=1")

    def test_alerts_rejects_anonymous(self, client):
        """Unauthenticated GET to /pharmacy/fefo/alerts must be rejected."""
        _assert_auth_required(client, "get", "/pharmacy/fefo/alerts")


# ─── Oncology chemo ──────────────────────────────────────────────────────────

class TestOncologyAuthRequired:
    """
    Chemotherapy dose calculation and order saving involve life-critical
    drug dosages.  These endpoints must require an authenticated clinician.
    """

    def test_chemo_builder_rejects_anonymous(self, client):
        _assert_auth_required(client, "get", "/medicine/oncology/chemo-builder/P-0001")

    def test_calculate_chemo_rejects_anonymous(self, client):
        _assert_auth_required(
            client, "get", "/medicine/oncology/api/calculate-chemo?patient_id=P-0001"
        )

    def test_save_chemo_order_rejects_anonymous(self, client):
        _assert_auth_required(
            client,
            "post",
            "/medicine/oncology/api/save-chemo-order",
            payload={"patient_id": "P-0001", "protocol_name": "AC-T"},
        )


# ─── Lab results ─────────────────────────────────────────────────────────────

class TestLabResultsAuthRequired:
    """
    Lab result routes return clinical data and must not be accessible
    without authentication.
    """

    def test_lab_patients_rejects_anonymous(self, client):
        _assert_auth_required(client, "get", "/medicine/lab_patients")

    def test_lab_results_rejects_anonymous(self, client):
        _assert_auth_required(client, "get", "/medicine/lab_results/R-9999")

    def test_pending_lab_patients_rejects_anonymous(self, client):
        _assert_auth_required(client, "get", "/medicine/pending_lab_patients")

    def test_patient_lab_results_rejects_anonymous(self, client):
        _assert_auth_required(client, "get", "/medicine/patient_lab_results/P-0001")
