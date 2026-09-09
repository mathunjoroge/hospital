import logging
from datetime import datetime, timezone

from flask import has_request_context
from flask_login import current_user
from sqlalchemy import event

from departments.models.admin import Log

logger = logging.getLogger(__name__)


def register_audit_listeners():
    """Register automatic audit logging event listeners on key models."""
    from departments.models.billing import (
        Billing,
        DrugsBill,
        ImagingBill,
        LabBill,
        PaidBill,
    )
    from departments.models.laboratory import LabResult
    from departments.models.medicine import Imaging, PrescribedMedicine
    from departments.models.pharmacy import DispensedDrug
    from departments.models.records import Patient

    audited_models = [
        Patient,
        Billing,
        DrugsBill,
        PaidBill,
        LabBill,
        ImagingBill,
        DispensedDrug,
        LabResult,
        Imaging,
        PrescribedMedicine,
    ]

    for model in audited_models:

        @event.listens_for(model, "after_insert")
        def receive_after_insert(mapper, connection, target):
            user_id = None
            if (
                has_request_context()
                and hasattr(current_user, "id")
                and getattr(current_user, "is_authenticated", False)
            ):
                user_id = current_user.id
            model_name = target.__class__.__name__
            rec_id = getattr(target, "id", getattr(target, "patient_id", "N/A"))
            msg = f"Audit [INSERT] {model_name} (ID: {rec_id})"
            connection.execute(
                Log.__table__.insert().values(
                    timestamp=datetime.now(timezone.utc),
                    level="INFO",
                    message=msg,
                    user_id=user_id,
                    source="audit",
                )
            )

        @event.listens_for(model, "after_update")
        def receive_after_update(mapper, connection, target):
            user_id = None
            if (
                has_request_context()
                and hasattr(current_user, "id")
                and getattr(current_user, "is_authenticated", False)
            ):
                user_id = current_user.id
            model_name = target.__class__.__name__
            rec_id = getattr(target, "id", getattr(target, "patient_id", "N/A"))
            msg = f"Audit [UPDATE] {model_name} (ID: {rec_id})"
            connection.execute(
                Log.__table__.insert().values(
                    timestamp=datetime.now(timezone.utc),
                    level="INFO",
                    message=msg,
                    user_id=user_id,
                    source="audit",
                )
            )

        @event.listens_for(model, "after_delete")
        def receive_after_delete(mapper, connection, target):
            user_id = None
            if (
                has_request_context()
                and hasattr(current_user, "id")
                and getattr(current_user, "is_authenticated", False)
            ):
                user_id = current_user.id
            model_name = target.__class__.__name__
            rec_id = getattr(target, "id", getattr(target, "patient_id", "N/A"))
            msg = f"Audit [DELETE] {model_name} (ID: {rec_id})"
            connection.execute(
                Log.__table__.insert().values(
                    timestamp=datetime.now(timezone.utc),
                    level="INFO",
                    message=msg,
                    user_id=user_id,
                    source="audit",
                )
            )
