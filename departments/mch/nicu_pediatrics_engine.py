"""
departments/mch/nicu_pediatrics_engine.py
──────────────────────────────────────────
Clinical Engine for NICU & Pediatrics Workstation:
1. APGAR 1/5/10 Minute Scoring & Resuscitation Risk Triage
2. Bhutani Neonatal Hyperbilirubinemia & Phototherapy Risk Nomogram
3. WHO / CDC Pediatric Growth Percentiles & Z-Score Calculator
"""

from typing import Dict, Optional

from departments.mch.models import (
    NeonatalApgarRecord,
    PediatricGrowthRecord,
    PhototherapyAssessmentRecord,
)
from extensions import db


class NicuPediatricsEngine:
    """Core domain logic engine for NICU, Neonatal Care, and Pediatric Growth Tracking."""

    # ---------------------------------------------------------------------------
    # APGAR Scoring Engine
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_apgar_score(
        patient_id: str,
        time_interval: str,
        appearance: int,
        pulse: int,
        grimace: int,
        activity: int,
        respiration: int,
        resuscitation_notes: Optional[str] = None,
        recorded_by: Optional[str] = None,
        encounter_id: Optional[int] = None,
    ) -> NeonatalApgarRecord:
        """
        Compute total APGAR score (0-10) and evaluate neonatal depression risk category.
        Score Tiers:
          - 7 to 10: Normal / Reassuring (routine care)
          - 4 to 6:  Moderately Depressed (tactile stimulation, airway clearing, O2)
          - 0 to 3:  Severely Depressed (immediate resuscitation & NICU protocol)
        """
        # Clamp inputs 0-2
        app_score = max(0, min(2, appearance))
        pulse_score = max(0, min(2, pulse))
        grim_score = max(0, min(2, grimace))
        act_score = max(0, min(2, activity))
        resp_score = max(0, min(2, respiration))

        total = app_score + pulse_score + grim_score + act_score + resp_score

        if total >= 7:
            risk = "NORMAL"
        elif total >= 4:
            risk = "MODERATE_DEPRESSION"
        else:
            risk = "SEVERE_DEPRESSION"

        record = NeonatalApgarRecord(
            patient_id=patient_id,
            encounter_id=encounter_id,
            time_interval=time_interval.upper(),
            appearance=app_score,
            pulse=pulse_score,
            grimace=grim_score,
            activity=act_score,
            respiration=resp_score,
            total_score=total,
            risk_category=risk,
            resuscitation_notes=resuscitation_notes,
            recorded_by=recorded_by or "Clinician",
        )
        db.session.add(record)
        db.session.commit()

        return record

    # ---------------------------------------------------------------------------
    # Bhutani Phototherapy Risk Nomogram
    # ---------------------------------------------------------------------------
    @staticmethod
    def evaluate_phototherapy_risk(
        patient_id: str,
        age_hours: int,
        serum_bilirubin_mg_dl: float,
        gestational_weeks: int = 38,
        has_hemolysis_risk: bool = False,
    ) -> PhototherapyAssessmentRecord:
        """
        Evaluate Total Serum Bilirubin (TSB) against postnatal age in hours (AAP / Bhutani guidelines).
        Identifies Phototherapy & Exchange Transfusion thresholds.
        """
        tsb = float(serum_bilirubin_mg_dl)
        hrs = max(4, int(age_hours))

        # Approximate Bhutani Nomogram 95th, 75th, 40th percentile threshold curves for TSB (mg/dL)
        if hrs <= 24:
            p95, p75, p40 = 8.0, 6.0, 4.5
            photo_cutoff = 8.0 if (gestational_weeks < 38 or has_hemolysis_risk) else 10.0
            exchange_cutoff = 15.0
        elif hrs <= 48:
            p95, p75, p40 = 13.0, 10.0, 7.5
            photo_cutoff = 11.0 if (gestational_weeks < 38 or has_hemolysis_risk) else 13.0
            exchange_cutoff = 19.0
        elif hrs <= 72:
            p95, p75, p40 = 16.0, 13.0, 10.0
            photo_cutoff = 13.0 if (gestational_weeks < 38 or has_hemolysis_risk) else 15.0
            exchange_cutoff = 22.0
        else:  # > 72h
            p95, p75, p40 = 17.5, 15.0, 12.0
            photo_cutoff = 15.0 if (gestational_weeks < 38 or has_hemolysis_risk) else 18.0
            exchange_cutoff = 25.0

        if tsb >= p95:
            risk_zone = "HIGH_RISK"
        elif tsb >= p75:
            risk_zone = "HIGH_INTERMEDIATE"
        elif tsb >= p40:
            risk_zone = "LOW_INTERMEDIATE"
        else:
            risk_zone = "LOW_RISK"

        photo_needed = tsb >= photo_cutoff
        exchange_needed = tsb >= exchange_cutoff

        if exchange_needed:
            rec = "CRITICAL: TSB at Exchange Transfusion threshold. Prepare NICU intensive phototherapy and IVIG / Exchange Transfusion."
        elif photo_needed:
            rec = "INITIATE PHOTOTHERAPY: TSB meets AAP phototherapy threshold for gestational age and risk factors. Recheck TSB in 4-8 hours."
        elif risk_zone in ("HIGH_RISK", "HIGH_INTERMEDIATE"):
            rec = "HIGH RE-CHECK RISK: Repeat TSB measurement in 6 to 12 hours. Ensure adequate hydration and lactation support."
        else:
            rec = "LOW RISK: Routine clinical follow-up. Repeat TSB in 24-48 hours if clinically jaundiced."

        record = PhototherapyAssessmentRecord(
            patient_id=patient_id,
            age_hours=hrs,
            serum_bilirubin_mg_dl=tsb,
            gestational_weeks=gestational_weeks,
            has_hemolysis_risk=has_hemolysis_risk,
            risk_zone=risk_zone,
            phototherapy_indicated=photo_needed,
            exchange_transfusion_indicated=exchange_needed,
            clinical_recommendation=rec,
        )
        db.session.add(record)
        db.session.commit()

        return record

    # ---------------------------------------------------------------------------
    # WHO / CDC Growth Z-Score Engine
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_growth_percentiles(
        patient_id: str,
        age_months: float,
        weight_kg: float,
        height_cm: Optional[float] = None,
        head_circumference_cm: Optional[float] = None,
        encounter_id: Optional[int] = None,
    ) -> PediatricGrowthRecord:
        """
        Calculate WHO growth Z-scores for Weight-for-Age, Height-for-Age, and Head Circumference.
        Formula: Z = ((measurement / M)^L - 1) / (L * S) (LMS Method approximation).
        """
        w = float(weight_kg)
        m = float(age_months)

        # Baseline Median Weight for Age (WHO standard reference approximation)
        median_weight = 3.3 + (0.75 * m) if m <= 12 else 10.0 + (0.2 * (m - 12))
        std_weight = 0.5 + (0.05 * m) if m <= 12 else 1.2 + (0.02 * (m - 12))

        weight_z = round((w - median_weight) / std_weight, 2)

        height_z = None
        if height_cm is not None:
            h = float(height_cm)
            median_height = 50.0 + (1.8 * m) if m <= 12 else 75.0 + (0.8 * (m - 12))
            std_height = 2.0 + (0.1 * m)
            height_z = round((h - median_height) / std_height, 2)

        head_z = None
        if head_circumference_cm is not None:
            hc = float(head_circumference_cm)
            median_hc = 35.0 + (0.8 * m) if m <= 12 else 45.0 + (0.1 * (m - 12))
            std_hc = 1.2 + (0.02 * m)
            head_z = round((hc - median_hc) / std_hc, 2)

        if weight_z < -3.0:
            status = "SEVERE_ACUTE_MALNUTRITION"
        elif weight_z < -2.0:
            status = "UNDERWEIGHT"
        elif height_z is not None and height_z < -2.0:
            status = "STUNTED"
        elif weight_z > 2.0:
            status = "OVERWEIGHT"
        else:
            status = "NORMAL"

        record = PediatricGrowthRecord(
            patient_id=patient_id,
            encounter_id=encounter_id,
            age_months=m,
            weight_kg=w,
            height_cm=height_cm,
            head_circumference_cm=head_circumference_cm,
            weight_for_age_zscore=weight_z,
            height_for_age_zscore=height_z,
            head_circ_zscore=head_z,
            nutritional_status=status,
        )
        db.session.add(record)
        db.session.commit()

        return record

    @staticmethod
    def get_nicu_workstation_summary(patient_id: str) -> Dict:
        """
        Aggregate complete NICU & Pediatric workstation profile for a patient.
        """
        apgars = (
            NeonatalApgarRecord.query.filter_by(patient_id=patient_id)
            .order_by(NeonatalApgarRecord.recorded_at.desc())
            .all()
        )
        bili_assessments = (
            PhototherapyAssessmentRecord.query.filter_by(patient_id=patient_id)
            .order_by(PhototherapyAssessmentRecord.recorded_at.desc())
            .all()
        )
        growth_records = (
            PediatricGrowthRecord.query.filter_by(patient_id=patient_id)
            .order_by(PediatricGrowthRecord.recorded_at.desc())
            .all()
        )

        latest_apgar = apgars[0] if apgars else None
        latest_bili = bili_assessments[0] if bili_assessments else None
        latest_growth = growth_records[0] if growth_records else None

        return {
            "patient_id": patient_id,
            "latest_apgar": {
                "interval": latest_apgar.time_interval,
                "score": latest_apgar.total_score,
                "risk": latest_apgar.risk_category,
                "breakdown": {
                    "appearance": latest_apgar.appearance,
                    "pulse": latest_apgar.pulse,
                    "grimace": latest_apgar.grimace,
                    "activity": latest_apgar.activity,
                    "respiration": latest_apgar.respiration,
                },
            }
            if latest_apgar
            else None,
            "latest_phototherapy": {
                "age_hours": latest_bili.age_hours,
                "tsb_mg_dl": latest_bili.serum_bilirubin_mg_dl,
                "risk_zone": latest_bili.risk_zone,
                "phototherapy_indicated": latest_bili.phototherapy_indicated,
                "exchange_indicated": latest_bili.exchange_transfusion_indicated,
                "recommendation": latest_bili.clinical_recommendation,
            }
            if latest_bili
            else None,
            "latest_growth": {
                "age_months": latest_growth.age_months,
                "weight_kg": latest_growth.weight_kg,
                "weight_z": latest_growth.weight_for_age_zscore,
                "height_z": latest_growth.height_for_age_zscore,
                "head_z": latest_growth.head_circ_zscore,
                "status": latest_growth.nutritional_status,
            }
            if latest_growth
            else None,
            "total_apgars": len(apgars),
            "total_bili_checks": len(bili_assessments),
            "total_growth_checks": len(growth_records),
        }
