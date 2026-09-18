"""
departments/icu/engine.py
─────────────────────────
Clinical Calculation Engine & Flowsheet Aggregator for ICU / HDU.
Handles MAP calculation, Glasgow Coma Scale (GCS) assessment,
I/O fluid balance summation, oliguria risk warnings, and trend matrices.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from departments.models.icu import ICUFlowsheetEntry, ICUFluidBalance


def calculate_map(systolic: float | None, diastolic: float | None) -> float | None:
    """
    Calculate Mean Arterial Pressure (MAP).
    MAP = Diastolic + 1/3 * (Systolic - Diastolic)
    """
    if systolic is None or diastolic is None:
        return None
    try:
        s = float(systolic)
        d = float(diastolic)
        if s <= 0 or d <= 0 or s < d:
            return None
        return round(d + (s - d) / 3.0, 1)
    except (ValueError, TypeError):
        return None


def calculate_gcs(
    eye: int | None, verbal: int | None, motor: int | None
) -> dict[str, Any]:
    """
    Calculate Glasgow Coma Scale (GCS) score and severity classification.
    Eye: 1-4, Verbal: 1-5, Motor: 1-6. Total: 3-15.
    """
    if eye is None or verbal is None or motor is None:
        return {
            "total": None,
            "severity": "Incomplete Assessment",
            "valid": False,
        }

    e = max(1, min(4, int(eye)))
    v = max(1, min(5, int(verbal)))
    m = max(1, min(6, int(motor)))
    total = e + v + m

    if total <= 8:
        severity = "Severe Head Injury / Coma (Airway Protection Required)"
    elif total <= 12:
        severity = "Moderate Head Injury"
    else:
        severity = "Mild / Normal Consciousness"

    return {
        "eye": e,
        "verbal": v,
        "motor": m,
        "total": total,
        "severity": severity,
        "valid": True,
    }


def calculate_fluid_balance(
    iv_fluids: float = 0.0,
    blood_products: float = 0.0,
    enteral: float = 0.0,
    medications: float = 0.0,
    urine: float = 0.0,
    drains: float = 0.0,
    ng_emesis: float = 0.0,
    stool: float = 0.0,
    weight_kg: float = 70.0,
    period_hours: float = 1.0,
) -> dict[str, Any]:
    """
    Calculate Input/Output fluid balance and urine output rate (mL/kg/hr).
    Generates oliguria alert if urine output rate < 0.5 mL/kg/hr.
    """
    total_input = round(
        float(iv_fluids) + float(blood_products) + float(enteral) + float(medications),
        1,
    )
    total_output = round(
        float(urine) + float(drains) + float(ng_emesis) + float(stool), 1
    )
    net_balance = round(total_input - total_output, 1)

    w = float(weight_kg) if weight_kg and float(weight_kg) > 0 else 70.0
    hrs = float(period_hours) if period_hours and float(period_hours) > 0 else 1.0

    urine_rate = round(float(urine) / (w * hrs), 2)
    is_oliguria = (
        urine_rate < 0.5 and float(urine) > 0
    )  # Alert if urine recorded and low

    return {
        "total_input_ml": total_input,
        "total_output_ml": total_output,
        "net_balance_ml": net_balance,
        "urine_rate_ml_kg_hr": urine_rate,
        "is_oliguria": is_oliguria,
        "oliguria_warning": "CRITICAL: Urine output < 0.5 mL/kg/hr (Oliguria / AKI Risk)"
        if is_oliguria
        else None,
    }


def generate_flowsheet_matrix(patient_id: str, hours: int = 24) -> dict[str, Any]:
    """
    Generate chronological trend matrix for ICU flowsheet visualization over the past `hours`.
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

    vitals_entries = (
        ICUFlowsheetEntry.query.filter(
            ICUFlowsheetEntry.patient_id == patient_id,
            ICUFlowsheetEntry.timestamp >= cutoff,
        )
        .order_by(ICUFlowsheetEntry.timestamp.asc())
        .all()
    )

    fluid_entries = (
        ICUFluidBalance.query.filter(
            ICUFluidBalance.patient_id == patient_id,
            ICUFluidBalance.timestamp >= cutoff,
        )
        .order_by(ICUFluidBalance.timestamp.asc())
        .all()
    )

    # Calculate 24h Totals
    total_24h_input = sum(f.total_input_ml for f in fluid_entries)
    total_24h_output = sum(f.total_output_ml for f in fluid_entries)
    net_24h_balance = round(total_24h_input - total_24h_output, 1)

    # Latest GCS
    latest_gcs = None
    for v in reversed(vitals_entries):
        if v.gcs_total is not None:
            gcs_eval = calculate_gcs(v.gcs_eye, v.gcs_verbal, v.gcs_motor)
            latest_gcs = {
                "score": v.gcs_total,
                "eye": v.gcs_eye,
                "verbal": v.gcs_verbal,
                "motor": v.gcs_motor,
                "severity": gcs_eval["severity"],
                "timestamp": v.timestamp.strftime("%Y-%m-%d %H:%M"),
            }
            break

    # Latest Vitals & Ventilator summary
    latest_vitals = None
    if vitals_entries:
        lv = vitals_entries[-1]
        latest_vitals = {
            "heart_rate": lv.heart_rate,
            "bp": f"{lv.bp_systolic}/{lv.bp_diastolic}"
            if lv.bp_systolic and lv.bp_diastolic
            else None,
            "map": lv.mean_arterial_pressure,
            "spo2": lv.spo2,
            "temp": lv.temperature,
            "cvp": lv.central_venous_pressure,
            "vent_mode": lv.ventilator_mode,
            "fio2": lv.fio2,
            "peep": lv.peep,
            "tidal_volume": lv.tidal_volume,
            "pip": lv.peak_inspiratory_pressure,
            "timestamp": lv.timestamp.strftime("%Y-%m-%d %H:%M"),
        }

    return {
        "patient_id": patient_id,
        "hours": hours,
        "vitals_entries": [
            {
                "id": v.id,
                "time": v.timestamp.strftime("%H:%M"),
                "timestamp": v.timestamp.strftime("%Y-%m-%d %H:%M"),
                "heart_rate": v.heart_rate,
                "bp_systolic": v.bp_systolic,
                "bp_diastolic": v.bp_diastolic,
                "map": v.mean_arterial_pressure
                or calculate_map(v.bp_systolic, v.bp_diastolic),
                "spo2": v.spo2,
                "temperature": v.temperature,
                "cvp": v.central_venous_pressure,
                "ventilator_mode": v.ventilator_mode,
                "fio2": v.fio2,
                "peep": v.peep,
                "tidal_volume": v.tidal_volume,
                "pip": v.peak_inspiratory_pressure,
                "respiratory_rate": v.respiratory_rate,
                "gcs_total": v.gcs_total,
                "rass_score": v.rass_score,
                "pain_score": v.pain_score,
                "notes": v.notes,
            }
            for v in vitals_entries
        ],
        "fluid_entries": [
            {
                "id": f.id,
                "time": f.timestamp.strftime("%H:%M"),
                "timestamp": f.timestamp.strftime("%Y-%m-%d %H:%M"),
                "iv_fluids": f.iv_fluids_ml,
                "blood_products": f.blood_products_ml,
                "enteral": f.enteral_oral_ml,
                "medications": f.iv_medications_ml,
                "total_input": f.total_input_ml,
                "urine": f.urine_output_ml,
                "drains": f.drain_output_ml,
                "ng_emesis": f.ng_emesis_ml,
                "stool": f.stool_ml,
                "total_output": f.total_output_ml,
                "net_balance": f.net_balance_ml,
                "notes": f.notes,
            }
            for f in fluid_entries
        ],
        "summary": {
            "total_24h_input_ml": total_24h_input,
            "total_24h_output_ml": total_24h_output,
            "net_24h_balance_ml": net_24h_balance,
            "latest_gcs": latest_gcs,
            "latest_vitals": latest_vitals,
        },
    }
