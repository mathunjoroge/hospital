import json
import random
import string
from datetime import datetime, timezone

from departments.models.laboratory import LabQCResult, LabQCSample, Specimen
from extensions import db


class WestgardEngine:
    """Westgard Multi-Rule Quality Control (QC) Engine for Clinical Analyzers.

    Evaluates:
    - 1_2s: Single control result exceeds Mean ± 2SD (Warning)
    - 1_3s: Single control result exceeds Mean ± 3SD (Reject)
    - 2_2s: 2 consecutive results exceed Mean + 2SD or Mean - 2SD (Reject)
    - R_4s: Difference between 2 control results in same run or consecutive exceeds 4SD (Reject)
    - 4_1s: 4 consecutive results exceed Mean + 1SD or Mean - 1SD (Reject)
    - 10_x: 10 consecutive results lie on same side of mean (Z > 0 or Z < 0) (Reject)
    """

    REJECTION_RULES = {"1_3s", "2_2s", "R_4s", "4_1s", "10_x"}
    WARNING_RULES = {"1_2s"}

    @staticmethod
    def calculate_z_score(measured_value: float, mean: float, sd: float) -> float:
        if sd <= 0:
            return 0.0
        return round((measured_value - mean) / sd, 4)

    @classmethod
    def evaluate_qc_run(
        cls, measured_value: float, mean: float, sd: float, history_z_scores: list[float]
    ) -> tuple[str, float, list[str]]:
        """Evaluates measured value against Westgard rules given historical Z-scores (most recent first).

        Returns:
            (status: "PASS" | "WARNING" | "REJECT", z_score: float, violated_rules: list[str])
        """
        z_score = cls.calculate_z_score(measured_value, mean, sd)
        z_series = [z_score] + list(history_z_scores)  # Index 0 is current run

        violated_rules = []

        # 1_3s rule: |Z| > 3.0
        if abs(z_score) > 3.0:
            violated_rules.append("1_3s")

        # 1_2s rule: |Z| > 2.0 (Warning trigger)
        if abs(z_score) > 2.0:
            violated_rules.append("1_2s")

        # 2_2s rule: 2 consecutive results > +2.0 or both < -2.0
        if len(z_series) >= 2:
            if (z_series[0] > 2.0 and z_series[1] > 2.0) or (
                z_series[0] < -2.0 and z_series[1] < -2.0
            ):
                violated_rules.append("2_2s")

        # R_4s rule: Difference between current and previous Z-score exceeds 4.0 SD (e.g. +2.1 and -2.1)
        if len(z_series) >= 2:
            if abs(z_series[0] - z_series[1]) >= 4.0:
                violated_rules.append("R_4s")

        # 4_1s rule: 4 consecutive results > +1.0 or all < -1.0
        if len(z_series) >= 4:
            last_4 = z_series[:4]
            if all(z > 1.0 for z in last_4) or all(z < -1.0 for z in last_4):
                violated_rules.append("4_1s")

        # 10_x rule: 10 consecutive results on same side of mean (Z > 0 or Z < 0)
        if len(z_series) >= 10:
            last_10 = z_series[:10]
            if all(z > 0.0 for z in last_10) or all(z < 0.0 for z in last_10):
                violated_rules.append("10_x")

        # Determine overall status
        triggered_rejections = set(violated_rules).intersection(cls.REJECTION_RULES)
        if triggered_rejections:
            status = "REJECT"
        elif "1_2s" in violated_rules:
            status = "WARNING"
        else:
            status = "PASS"

        return status, z_score, violated_rules


class LIMSService:
    """Enterprise LIMS Service for Specimen Lifecycle Tracking & QC."""

    @staticmethod
    def generate_barcode() -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        rand_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        return f"SPEC-{date_str}-{rand_str}"

    @classmethod
    def create_specimen(
        cls,
        patient_id: str,
        requested_lab_id: int | None = None,
        specimen_type: str = "WHOLE_BLOOD",
        container_type: str = "EDTA_PURPLE",
        user_id: int | None = None,
    ) -> Specimen:
        barcode = cls.generate_barcode()
        while Specimen.query.filter_by(barcode=barcode).first():
            barcode = cls.generate_barcode()

        now_iso = datetime.now(timezone.utc).isoformat()
        initial_coc = [
            {
                "timestamp": now_iso,
                "status": "ORDERED",
                "action": "Specimen ordered",
                "user_id": user_id,
            }
        ]

        specimen = Specimen(
            barcode=barcode,
            patient_id=patient_id,
            requested_lab_id=requested_lab_id,
            specimen_type=specimen_type,
            container_type=container_type,
            status="ORDERED",
            chain_of_custody=json.dumps(initial_coc),
        )
        db.session.add(specimen)
        db.session.commit()
        return specimen

    @classmethod
    def update_specimen_status(
        cls,
        specimen_id_or_barcode: str | int,
        new_status: str,
        user_id: int | None = None,
        notes: str | None = None,
        rejection_reason: str | None = None,
    ) -> Specimen:
        if isinstance(specimen_id_or_barcode, int) or specimen_id_or_barcode.isdigit():
            specimen = Specimen.query.get(int(specimen_id_or_barcode))
        else:
            specimen = Specimen.query.filter_by(barcode=specimen_id_or_barcode).first()

        if not specimen:
            raise ValueError(f"Specimen not found: {specimen_id_or_barcode}")

        now = datetime.now(timezone.utc)
        specimen.status = new_status

        if new_status == "COLLECTED":
            specimen.collected_at = now
            specimen.collected_by_id = user_id
        elif new_status == "RECEIVED":
            specimen.received_at = now
        elif new_status == "REJECTED":
            specimen.rejection_reason = rejection_reason or "Unspecified rejection"

        # Update chain of custody
        coc = json.loads(specimen.chain_of_custody) if specimen.chain_of_custody else []
        coc.append(
            {
                "timestamp": now.isoformat(),
                "status": new_status,
                "action": f"Status updated to {new_status}",
                "user_id": user_id,
                "notes": notes,
                "rejection_reason": rejection_reason,
            }
        )
        specimen.chain_of_custody = json.dumps(coc)

        db.session.commit()
        return specimen

    @classmethod
    def log_qc_result(
        cls, qc_sample_id: int, measured_value: float, operator_id: int | None = None
    ) -> LabQCResult:
        qc_sample = LabQCSample.query.get(qc_sample_id)
        if not qc_sample:
            raise ValueError(f"Lab QC sample not found: {qc_sample_id}")

        # Fetch recent Z-scores for this sample (ordered by timestamp desc)
        recent_results = (
            LabQCResult.query.filter_by(qc_sample_id=qc_sample_id)
            .order_by(LabQCResult.run_timestamp.desc())
            .limit(15)
            .all()
        )
        history_z_scores = [r.z_score for r in recent_results]

        status, z_score, violated_rules = WestgardEngine.evaluate_qc_run(
            measured_value=measured_value,
            mean=qc_sample.target_mean,
            sd=qc_sample.target_sd,
            history_z_scores=history_z_scores,
        )

        qc_result = LabQCResult(
            qc_sample_id=qc_sample_id,
            run_timestamp=datetime.now(timezone.utc),
            measured_value=measured_value,
            z_score=z_score,
            status=status,
            violated_rules=json.dumps(violated_rules),
            operator_id=operator_id,
        )
        db.session.add(qc_result)
        db.session.commit()
        return qc_result

    @classmethod
    def get_lims_dashboard_metrics(cls) -> dict:
        total_specimens = Specimen.query.count()
        status_counts = {}
        for st in [
            "ORDERED",
            "COLLECTED",
            "RECEIVED",
            "IN_ANALYSIS",
            "COMPLETED",
            "REJECTED",
            "DISPOSED",
        ]:
            status_counts[st.lower()] = Specimen.query.filter_by(status=st).count()

        total_qc_runs = LabQCResult.query.count()
        pass_qc_runs = LabQCResult.query.filter_by(status="PASS").count()
        warning_qc_runs = LabQCResult.query.filter_by(status="WARNING").count()
        reject_qc_runs = LabQCResult.query.filter_by(status="REJECT").count()

        pass_rate = round((pass_qc_runs / total_qc_runs * 100), 1) if total_qc_runs > 0 else 100.0

        return {
            "total_specimens": total_specimens,
            "status_counts": status_counts,
            "qc_metrics": {
                "total_runs": total_qc_runs,
                "pass_runs": pass_qc_runs,
                "warning_runs": warning_qc_runs,
                "reject_runs": reject_qc_runs,
                "pass_rate_pct": pass_rate,
            },
        }
