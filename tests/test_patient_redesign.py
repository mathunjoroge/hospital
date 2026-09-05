"""
tests/test_patient_redesign.py
──────────────────────────────
Unit tests for Task 2.1: Patient model redesign — soft-delete, audit
timestamps, nullable optional fields, PatientIdentifier, PatientMerge,
duplicate detection, and the merge helper.
"""
from datetime import date, datetime

import pytest

from departments.models.records import Patient, PatientIdentifier, PatientMerge
from departments.models.user import User
from departments.records.merge import find_duplicate_candidates, merge_patient_records
from extensions import db

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_patient(suffix, dob=None, contact=None, name=None, national_id=None):
    return Patient(
        patient_id=f"PT{suffix}",
        name=name or f"Patient {suffix}",
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=dob or date(1990, 1, 1),
        marital_status="Single",
        contact=contact or f"070000{suffix}",
        next_of_kin="Next Kin",
        relationship_with_next_of_kin="Parent",
        next_of_kin_contact="0711111111",
        national_id=national_id,
        emergency_contact="0722222222",
    )


def _make_user():
    from werkzeug.security import generate_password_hash
    user = User(username="teststaff", role="records",
                password=generate_password_hash("TestPass1!"))
    db.session.add(user)
    db.session.commit()
    return user


# ─────────────────────────────────────────────
# 1. nullable fields — national_id & blood_group
# ─────────────────────────────────────────────

class TestNullableFields:
    def test_register_without_national_id(self, app):
        """Patient can be registered with national_id=None (nullable)."""
        with app.app_context():
            p = _make_patient("N001", national_id=None)
            db.session.add(p)
            db.session.commit()
            saved = Patient.query.filter_by(patient_id="PTN001").first()
            assert saved is not None
            assert saved.national_id is None

    def test_register_without_blood_group(self, app):
        """Patient can be registered with blood_group=None (nullable)."""
        with app.app_context():
            p = _make_patient("N002")
            p.blood_group = None
            db.session.add(p)
            db.session.commit()
            saved = Patient.query.filter_by(patient_id="PTN002").first()
            assert saved.blood_group is None

    def test_register_with_optional_demographic_fields(self, app):
        """insurance_provider, occupation, employer_name store correctly."""
        with app.app_context():
            p = _make_patient("N003")
            p.insurance_provider = "SHA"
            p.insurance_policy_number = "SHA-9999"
            p.occupation = "Teacher"
            p.employer_name = "County Schools"
            db.session.add(p)
            db.session.commit()
            saved = Patient.query.filter_by(patient_id="PTN003").first()
            assert saved.insurance_provider == "SHA"
            assert saved.occupation == "Teacher"


# ─────────────────────────────────────────────
# 2. Soft-delete
# ─────────────────────────────────────────────

class TestSoftDelete:
    def test_soft_delete_sets_is_active_false(self, app):
        with app.app_context():
            p = _make_patient("SD01")
            db.session.add(p)
            db.session.commit()
            p.soft_delete()
            db.session.commit()
            saved = Patient.query.filter_by(patient_id="PTSD01").first()
            assert saved.is_active is False
            assert saved.deleted_at is not None

    def test_active_patients_excludes_soft_deleted(self, app):
        with app.app_context():
            active = _make_patient("SD02")
            deleted = _make_patient("SD03")
            db.session.add_all([active, deleted])
            db.session.commit()
            deleted.soft_delete()
            db.session.commit()
            actives = Patient.active_patients().all()
            ids = [p.patient_id for p in actives]
            assert "PTSD02" in ids
            assert "PTSD03" not in ids

    def test_soft_deleted_patient_still_in_db(self, app):
        """Soft-deleted row is still retrievable via unconstrained query."""
        with app.app_context():
            p = _make_patient("SD04")
            db.session.add(p)
            db.session.commit()
            p.soft_delete()
            db.session.commit()
            raw = Patient.query.filter_by(patient_id="PTSD04").first()
            assert raw is not None


# ─────────────────────────────────────────────
# 3. Audit timestamps
# ─────────────────────────────────────────────

class TestAuditTimestamps:
    def test_created_at_populated(self, app):
        with app.app_context():
            p = _make_patient("AT01")
            db.session.add(p)
            db.session.commit()
            assert p.created_at is not None
            assert isinstance(p.created_at, datetime)

    def test_updated_at_populated(self, app):
        with app.app_context():
            p = _make_patient("AT02")
            db.session.add(p)
            db.session.commit()
            assert p.updated_at is not None


# ─────────────────────────────────────────────
# 4. PatientIdentifier
# ─────────────────────────────────────────────

class TestPatientIdentifier:
    def test_create_alternate_identifier(self, app):
        with app.app_context():
            p = _make_patient("PI01")
            db.session.add(p)
            db.session.commit()
            ident = PatientIdentifier(
                patient_id="PTPI01",
                identifier_type="Passport",
                identifier_value="AB123456"
            )
            db.session.add(ident)
            db.session.commit()
            saved = PatientIdentifier.query.filter_by(
                patient_id="PTPI01", identifier_type="Passport"
            ).first()
            assert saved is not None
            assert saved.identifier_value == "AB123456"

    def test_multiple_identifiers_per_patient(self, app):
        with app.app_context():
            p = _make_patient("PI02")
            db.session.add(p)
            db.session.commit()
            for itype, ival in [
                ("Birth Certificate", "BC-001"),
                ("Refugee ID", "RF-999"),
            ]:
                db.session.add(PatientIdentifier(
                    patient_id="PTPI02",
                    identifier_type=itype,
                    identifier_value=ival
                ))
            db.session.commit()
            idents = PatientIdentifier.query.filter_by(patient_id="PTPI02").all()
            assert len(idents) == 2


# ─────────────────────────────────────────────
# 5. Duplicate detection
# ─────────────────────────────────────────────

class TestDuplicateDetection:
    def test_high_confidence_phone_dob_match(self, app):
        """Same DOB and last 9 digits of phone → HIGH confidence."""
        with app.app_context():
            p1 = _make_patient("DD01", dob=date(1985, 3, 10), contact="0712345678")
            p2 = _make_patient("DD02", dob=date(1985, 3, 10), contact="+254712345678")
            db.session.add_all([p1, p2])
            db.session.commit()
            result = find_duplicate_candidates(p1)
            assert any(r['confidence'] == 'HIGH' for r in result)

    def test_medium_confidence_name_dob_match(self, app):
        """Same DOB + shared name token → MEDIUM confidence."""
        with app.app_context():
            p1 = _make_patient("DD03", dob=date(1992, 7, 20), name="John Kamau", contact="0700000001")
            p2 = _make_patient("DD04", dob=date(1992, 7, 20), name="John Mwangi", contact="0700000002")
            db.session.add_all([p1, p2])
            db.session.commit()
            result = find_duplicate_candidates(p1)
            assert any(r['confidence'] == 'MEDIUM' for r in result)

    def test_no_match_different_dob(self, app):
        """Patients with different DOBs produce no candidates."""
        with app.app_context():
            p1 = _make_patient("DD05", dob=date(1980, 1, 1), contact="0700000101")
            p2 = _make_patient("DD06", dob=date(1990, 6, 15), contact="0700000102")
            db.session.add_all([p1, p2])
            db.session.commit()
            result = find_duplicate_candidates(p1)
            assert result == []

    def test_inactive_patient_excluded_from_candidates(self, app):
        """Soft-deleted patients do not appear as duplicate candidates."""
        with app.app_context():
            p1 = _make_patient("DD07", dob=date(1970, 5, 5), contact="0700000201")
            p2 = _make_patient("DD08", dob=date(1970, 5, 5), contact="0700000201")
            db.session.add_all([p1, p2])
            db.session.commit()
            p2.soft_delete()
            db.session.commit()
            result = find_duplicate_candidates(p1)
            assert all(r['patient'].is_active for r in result)


# ─────────────────────────────────────────────
# 6. Patient merge handler
# ─────────────────────────────────────────────

class TestPatientMerge:
    def test_merge_soft_deletes_source(self, app):
        with app.app_context():
            user = _make_user()
            src = _make_patient("MG01")
            tgt = _make_patient("MG02")
            db.session.add_all([src, tgt])
            db.session.commit()
            merge_patient_records("PTMG01", "PTMG02", user_id=user.id, notes="test merge")
            src_db = Patient.query.filter_by(patient_id="PTMG01").first()
            assert src_db.is_active is False
            assert src_db.deleted_at is not None

    def test_merge_target_remains_active(self, app):
        with app.app_context():
            user = _make_user()
            src = _make_patient("MG03")
            tgt = _make_patient("MG04")
            db.session.add_all([src, tgt])
            db.session.commit()
            merge_patient_records("PTMG03", "PTMG04", user_id=user.id)
            tgt_db = Patient.query.filter_by(patient_id="PTMG04").first()
            assert tgt_db.is_active is True

    def test_merge_creates_audit_record(self, app):
        with app.app_context():
            user = _make_user()
            src = _make_patient("MG05")
            tgt = _make_patient("MG06")
            db.session.add_all([src, tgt])
            db.session.commit()
            merge_patient_records("PTMG05", "PTMG06", user_id=user.id, notes="audit check")
            audit = PatientMerge.query.filter_by(
                source_patient_id="PTMG05",
                target_patient_id="PTMG06"
            ).first()
            assert audit is not None
            assert audit.notes == "audit check"

    def test_merge_already_inactive_source_raises(self, app):
        """Merging an already-merged/inactive patient must raise ValueError."""
        with app.app_context():
            user = _make_user()
            src = _make_patient("MG07")
            tgt = _make_patient("MG08")
            db.session.add_all([src, tgt])
            db.session.commit()
            src.soft_delete()
            db.session.commit()
            with pytest.raises(ValueError, match="already inactive"):
                merge_patient_records("PTMG07", "PTMG08", user_id=user.id)
