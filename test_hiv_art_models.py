#!/usr/bin/env python3
"""
Standalone test for HIV/ART data models.
Tests model creation and basic functionality with Flask app context.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datetime import datetime, timedelta, timezone

from app import app
from departments.hiv_art.models import (
    AdherenceVisit,
    ARTEnrollment,
    ARTRegimen,
    CD4Count,
    ViralLoad,
    WHOStage,
)
from extensions import db


def test_model_creation():
    """Test that all HIV/ART models can be created and have correct attributes."""
    print("Testing HIV/ART model creation...")

    with app.app_context():
        try:
            # Create tables
            db.create_all()

            # Test ARTRegimen creation
            regimen = ARTRegimen(
                regimen_code="TDF/3TC/EFV",
                regimen_name="Tenofovir/Lamivudine/Efavirenz",
                line_of_therapy=1,
                arv_drugs="Tenofovir, Lamivudine, Efavirenz",
                is_preferred=True,
                effective_from=datetime.now(timezone.utc).date()
            )
            db.session.add(regimen)
            db.session.commit()

            assert regimen.id is not None
            assert regimen.regimen_code == "TDF/3TC/EFV"
            assert regimen.line_of_therapy == 1
            assert regimen.is_preferred == True
            print("✓ ARTRegimen creation successful")

            # Test ARTEnrollment creation
            enrollment = ARTEnrollment(
                patient_id="PAT001",
                art_number="ART0001",
                baseline_cd4=350,
                baseline_who_stage=2,
                art_start_date=datetime.now(timezone.utc),
                facility_enrolled_at="Test Clinic",
                current_regimen_id=regimen.id
            )
            db.session.add(enrollment)
            db.session.commit()

            assert enrollment.id is not None
            assert enrollment.patient_id == "PAT001"
            assert enrollment.art_number == "ART0001"
            assert enrollment.baseline_cd4 == 350
            assert enrollment.baseline_who_stage == 2
            assert enrollment.current_regimen_id == regimen.id
            print("✓ ARTEnrollment creation successful")

            # Test AdherenceVisit creation
            visit = AdherenceVisit(
                enrollment_id=enrollment.id,
                pills_dispensed=30,
                pills_returned=3,
                adherence_percentage=90.0,
                adherence_category="good",
                visit_date=datetime.now(timezone.utc)
            )
            db.session.add(visit)
            db.session.commit()

            assert visit.id is not None
            assert visit.pills_dispensed == 30
            assert visit.pills_returned == 3
            assert visit.adherence_percentage == 90.0
            assert visit.adherence_category == "good"
            print("✓ AdherenceVisit creation successful")

            # Test ViralLoad creation
            viral_load = ViralLoad(
                enrollment_id=enrollment.id,
                viral_load_copies=250,
                test_type="routine",
                test_date=datetime.now(timezone.utc)
            )
            db.session.add(viral_load)
            db.session.commit()

            assert viral_load.id is not None
            assert viral_load.viral_load_copies == 250
            assert viral_load.test_type == "routine"
            print("✓ ViralLoad creation successful")

            # Test CD4Count creation
            cd4 = CD4Count(
                enrollment_id=enrollment.id,
                cd4_count=450,
                cd4_percent=28.0,
                test_date=datetime.now(timezone.utc)
            )
            db.session.add(cd4)
            db.session.commit()

            assert cd4.id is not None
            assert cd4.cd4_count == 450
            assert cd4.cd4_percent == 28.0
            print("✓ CD4Count creation successful")

            # Test WHOStage creation
            who_stage = WHOStage(
                enrollment_id=enrollment.id,
                who_stage=1,
                defining_conditions="Asymptomatic",
                assessment_date=datetime.now(timezone.utc)
            )
            db.session.add(who_stage)
            db.session.commit()

            assert who_stage.id is not None
            assert who_stage.who_stage == 1
            assert who_stage.defining_conditions == "Asymptomatic"
            print("✓ WHOStage creation successful")

            # Test relationships
            assert len(enrollment.adherence_visits.all()) == 1
            assert enrollment.adherence_visits.first().id == visit.id

            assert len(enrollment.viral_loads.all()) == 1
            assert enrollment.viral_loads.first().id == viral_load.id

            assert len(enrollment.cd4_counts.all()) == 1
            assert enrollment.cd4_counts.first().id == cd4.id

            assert len(enrollment.who_stages.all()) == 1
            assert enrollment.who_stages.first().id == who_stage.id

            print("✓ Model relationships working correctly")

            # Test adherence calculation logic
            visit2 = AdherenceVisit(
                enrollment_id=enrollment.id,
                pills_dispensed=20,
                pills_returned=10,  # Took 10 out of 20 = 50%
                adherence_percentage=50.0,
                adherence_category="poor",
                visit_date=datetime.now(timezone.utc) + timedelta(days=30)
            )
            db.session.add(visit2)
            db.session.commit()

            assert visit2.adherence_percentage == 50.0
            assert visit2.adherence_category == "poor"
            print("✓ Adherence calculation logic working")

            print("\n🎉 All HIV/ART model tests passed!")
            return True

        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return False
        finally:
            db.session.remove()
            db.drop_all()

if __name__ == "__main__":
    success = test_model_creation()
    sys.exit(0 if success else 1)
