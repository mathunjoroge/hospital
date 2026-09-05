import json
from datetime import datetime

from departments.models.laboratory import LabResult
from departments.models.medicine import LabTest, RequestedLab
from extensions import db


def test_lab_flow_request_to_result(app):
    with app.app_context():
        # 1. Create Lab Test
        lab_test = LabTest(
            test_name="Full Blood Count",
            cost=2000.0,
            description="Routine hematology test"
        )
        db.session.add(lab_test)
        db.session.commit()

        # 2. Patient Requests Lab
        lab_request = RequestedLab(
            patient_id="PTEST100",
            lab_test_id=lab_test.id,
            date_requested=datetime.utcnow(),
            status=0
        )
        db.session.add(lab_request)
        db.session.commit()

        assert lab_request.status == 0

        # 3. Process Lab Request -> Save Result
        results_data = json.dumps({"Hb": "14.2", "WBC": "6.5"})
        lab_result = LabResult(
            patient_id="PTEST100",
            lab_test_id=lab_test.id,
            test_date=datetime.utcnow(),
            result=results_data,
            result_notes="Normal blood parameters",
            result_id="RES-1001",
            updated_by=1
        )
        db.session.add(lab_result)
        lab_request.status = 1
        lab_request.result_id = "RES-1001"
        db.session.commit()

        # 4. Verify status updated and result saved
        updated_request = RequestedLab.query.get(lab_request.id)
        saved_result = LabResult.query.filter_by(result_id="RES-1001").first()

        assert updated_request.status == 1
        assert saved_result is not None
        assert json.loads(saved_result.result)["Hb"] == "14.2"
