"""
departments/mch/cold_chain.py
─────────────────────────────────────────────────────────────────────────────
Cold-Chain Monitoring & Vaccine Inventory Engine.

Aligns with WHO/KEPI cold-chain requirements:
  - Vaccine storage temperature band: +2 °C to +8 °C (refrigerator)
  - Freeze-sensitive vaccines must not fall below 0 °C
  - Temperature excursions trigger an alert and mark affected stock at-risk
  - VVM (Vaccine Vial Monitor) status tracking per batch
  - FEFO (First Expiry / First Out) dispensing order enforcement

References:
  - WHO EVM (Effective Vaccine Management) criteria
  - Kenya KEPI Cold-Chain Policy 2020
  - UNICEF Supply Division — Cold Chain Storage Guidelines
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from extensions import db

from .models import VaccineBatch, VaccineTemperatureLog

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

COLD_CHAIN_MIN_C: float = 2.0   # °C  — lower safe limit
COLD_CHAIN_MAX_C: float = 8.0   # °C  — upper safe limit
FREEZE_SENSITIVE_MIN_C: float = 0.0  # °C  — absolute lower for freeze-sensitive

FREEZE_SENSITIVE_VACCINES = {
    "DTP",
    "Pentavalent",
    "Hib",
    "Hepatitis B",
    "IPV",
    "PCV",
    "Rotavirus",
    "HPV",
}


# ── Engine ────────────────────────────────────────────────────────────────────


class ColdChainEngine:
    """
    Manages vaccine cold-chain logistics:
      - Stock receipt and expiry-aware inventory
      - Continuous temperature telemetry with breach detection
      - VVM stage tracking
      - FEFO-ordered dispensing
    """

    # ── Stock management ──────────────────────────────────────────────────────

    def receive_vaccine_batch(
        self,
        vaccine_name: str,
        batch_number: str,
        manufacturer: str,
        quantity: int,
        doses_per_vial: int,
        expiry_date,        # datetime.date
        storage_location: str,
        supplied_by: str | None = None,
    ) -> "VaccineBatch":
        """
        Records receipt of a new vaccine batch into cold-chain stock.

        Validates the batch isn't already registered and that expiry_date
        is in the future before persisting.
        """
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if doses_per_vial <= 0:
            raise ValueError("doses_per_vial must be > 0")

        today = datetime.now(timezone.utc).date()
        if expiry_date <= today:
            raise ValueError(f"Batch {batch_number} is already expired.")

        existing = VaccineBatch.query.filter_by(
            vaccine_name=vaccine_name,
            batch_number=batch_number,
        ).first()
        if existing:
            raise ValueError(
                f"Batch {batch_number} for {vaccine_name} is already registered."
            )

        batch = VaccineBatch(
            vaccine_name=vaccine_name,
            batch_number=batch_number,
            manufacturer=manufacturer,
            quantity_vials=quantity,
            doses_per_vial=doses_per_vial,
            quantity_remaining_vials=quantity,
            expiry_date=expiry_date,
            storage_location=storage_location,
            supplied_by=supplied_by,
            vvm_stage=1,          # VVM stage 1 = unused / OK
            is_cold_chain_breach=False,
        )
        db.session.add(batch)
        db.session.commit()

        logger.info(
            "COLD CHAIN RECEIPT: %s batch %s  qty=%d vials  exp=%s  loc=%s",
            vaccine_name,
            batch_number,
            quantity,
            expiry_date,
            storage_location,
        )
        return batch

    def dispense_vaccine(
        self,
        vaccine_name: str,
        vials_needed: int,
        immunization_record_id: str | None = None,
    ) -> list["VaccineBatch"]:
        """
        Dispenses vials from cold-chain stock using FEFO order.

        Returns the list of batches drawn from (for audit purposes).
        Raises ValueError if insufficient stock or no eligible batch found.
        """
        if vials_needed <= 0:
            raise ValueError("vials_needed must be > 0")

        today = datetime.now(timezone.utc).date()

        # FEFO: order by expiry_date ASC, exclude breached or expired batches
        available = (
            VaccineBatch.query.filter_by(
                vaccine_name=vaccine_name,
                is_cold_chain_breach=False,
            )
            .filter(
                VaccineBatch.expiry_date > today,
                VaccineBatch.quantity_remaining_vials > 0,
                VaccineBatch.vvm_stage <= 2,  # stages 3/4 = unusable
            )
            .order_by(VaccineBatch.expiry_date.asc())
            .all()
        )

        total_available = sum(b.quantity_remaining_vials for b in available)
        if total_available < vials_needed:
            raise ValueError(
                f"Insufficient cold-chain stock for {vaccine_name}: "
                f"need {vials_needed}, available {total_available} vials."
            )

        drawn_from = []
        remaining = vials_needed
        for batch in available:
            if remaining <= 0:
                break
            draw = min(batch.quantity_remaining_vials, remaining)
            batch.quantity_remaining_vials -= draw
            remaining -= draw
            drawn_from.append(batch)
            logger.info(
                "COLD CHAIN DISPENSE: %s batch %s  drew %d vials  (remaining=%d)",
                vaccine_name,
                batch.batch_number,
                draw,
                batch.quantity_remaining_vials,
            )

        db.session.commit()
        return drawn_from

    # ── Temperature monitoring ────────────────────────────────────────────────

    def log_temperature(
        self,
        storage_location: str,
        temperature_celsius: float,
        logged_by: str | None = None,
        sensor_id: str | None = None,
        notes: str | None = None,
    ) -> "VaccineTemperatureLog":
        """
        Records a temperature reading for a storage location and detects breaches.

        A breach is flagged when temperature falls outside [2 °C, 8 °C].
        Freeze-sensitive batches stored at that location are automatically
        marked as at-risk when temperature drops below 0 °C.
        """
        is_breach = not (COLD_CHAIN_MIN_C <= temperature_celsius <= COLD_CHAIN_MAX_C)
        breach_type = None

        if temperature_celsius < COLD_CHAIN_MIN_C:
            breach_type = "FREEZE" if temperature_celsius < FREEZE_SENSITIVE_MIN_C else "TOO_COLD"
        elif temperature_celsius > COLD_CHAIN_MAX_C:
            breach_type = "TOO_HOT"

        log_entry = VaccineTemperatureLog(
            storage_location=storage_location,
            temperature_celsius=temperature_celsius,
            is_breach=is_breach,
            breach_type=breach_type,
            logged_by=logged_by,
            sensor_id=sensor_id,
            notes=notes,
        )
        db.session.add(log_entry)

        # Automatically escalate VVM and mark freeze-sensitive batches at-risk
        if breach_type == "FREEZE":
            affected = VaccineBatch.query.filter_by(
                storage_location=storage_location,
                is_cold_chain_breach=False,
            ).filter(VaccineBatch.vaccine_name.in_(FREEZE_SENSITIVE_VACCINES)).all()

            for batch in affected:
                batch.is_cold_chain_breach = True
                batch.vvm_stage = 3  # Compromised
                logger.warning(
                    "COLD CHAIN BREACH (FREEZE): batch %s (%s) at %s marked compromised  T=%.1f°C",
                    batch.batch_number,
                    batch.vaccine_name,
                    storage_location,
                    temperature_celsius,
                )

        elif breach_type == "TOO_HOT":
            affected = VaccineBatch.query.filter_by(
                storage_location=storage_location,
                is_cold_chain_breach=False,
            ).filter(VaccineBatch.quantity_remaining_vials > 0).all()

            for batch in affected:
                # Escalate VVM stage — heat exposure degrades the monitor
                if batch.vvm_stage < 3:
                    batch.vvm_stage = min(batch.vvm_stage + 1, 4)
                    if batch.vvm_stage >= 3:
                        batch.is_cold_chain_breach = True
                        logger.warning(
                            "COLD CHAIN BREACH (HEAT): batch %s (%s) VVM→%d — marked compromised",
                            batch.batch_number,
                            batch.vaccine_name,
                            batch.vvm_stage,
                        )

        db.session.commit()

        if is_breach:
            logger.warning(
                "TEMPERATURE BREACH at %s: %.1f°C (%s) — sensor=%s",
                storage_location,
                temperature_celsius,
                breach_type,
                sensor_id,
            )
        else:
            logger.debug(
                "Temperature OK at %s: %.1f°C",
                storage_location,
                temperature_celsius,
            )

        return log_entry

    # ── Reporting helpers ─────────────────────────────────────────────────────

    def get_stock_summary(self, vaccine_name: Optional[str] = None) -> list[dict]:
        """
        Returns current cold-chain stock, optionally filtered by vaccine.
        Includes FEFO-sorted remaining inventory and breach flags.
        """
        today = datetime.now(timezone.utc).date()
        query = VaccineBatch.query

        if vaccine_name:
            query = query.filter_by(vaccine_name=vaccine_name)

        batches = query.order_by(
            VaccineBatch.vaccine_name.asc(),
            VaccineBatch.expiry_date.asc(),
        ).all()

        result = []
        for b in batches:
            days_to_expiry = (b.expiry_date - today).days
            result.append({
                "id": b.id,
                "vaccine_name": b.vaccine_name,
                "batch_number": b.batch_number,
                "manufacturer": b.manufacturer,
                "quantity_vials": b.quantity_vials,
                "quantity_remaining_vials": b.quantity_remaining_vials,
                "total_doses_remaining": b.quantity_remaining_vials * b.doses_per_vial,
                "expiry_date": b.expiry_date.isoformat(),
                "days_to_expiry": days_to_expiry,
                "storage_location": b.storage_location,
                "vvm_stage": b.vvm_stage,
                "is_cold_chain_breach": b.is_cold_chain_breach,
                "status": self._batch_status(b, days_to_expiry),
            })
        return result

    def get_temperature_history(
        self,
        storage_location: str,
        limit: int = 100,
        breaches_only: bool = False,
    ) -> list[dict]:
        """Returns recent temperature logs for a given storage location."""
        query = VaccineTemperatureLog.query.filter_by(storage_location=storage_location)
        if breaches_only:
            query = query.filter_by(is_breach=True)

        logs = query.order_by(VaccineTemperatureLog.recorded_at.desc()).limit(limit).all()
        return [
            {
                "id": log.id,
                "storage_location": log.storage_location,
                "temperature_celsius": log.temperature_celsius,
                "is_breach": log.is_breach,
                "breach_type": log.breach_type,
                "recorded_at": log.recorded_at.isoformat(),
                "sensor_id": log.sensor_id,
                "logged_by": log.logged_by,
                "notes": log.notes,
            }
            for log in logs
        ]

    def get_near_expiry_alerts(self, days_threshold: int = 30) -> list[dict]:
        """Returns vaccine batches expiring within `days_threshold` days."""
        today = datetime.now(timezone.utc).date()
        batches = (
            VaccineBatch.query.filter(
                VaccineBatch.quantity_remaining_vials > 0,
                VaccineBatch.expiry_date > today,
            )
            .order_by(VaccineBatch.expiry_date.asc())
            .all()
        )
        alerts = []
        for b in batches:
            days = (b.expiry_date - today).days
            if days <= days_threshold:
                alerts.append({
                    "vaccine_name": b.vaccine_name,
                    "batch_number": b.batch_number,
                    "expiry_date": b.expiry_date.isoformat(),
                    "days_to_expiry": days,
                    "quantity_remaining_vials": b.quantity_remaining_vials,
                    "storage_location": b.storage_location,
                })
        return alerts

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _batch_status(batch: "VaccineBatch", days_to_expiry: int) -> str:
        if batch.is_cold_chain_breach or batch.vvm_stage >= 3:
            return "COMPROMISED"
        if days_to_expiry <= 0:
            return "EXPIRED"
        if days_to_expiry <= 30:
            return "NEAR_EXPIRY"
        if batch.quantity_remaining_vials == 0:
            return "DEPLETED"
        return "OK"
