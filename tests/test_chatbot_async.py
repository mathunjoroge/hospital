from unittest.mock import MagicMock, patch

import pytest
from werkzeug.security import generate_password_hash

from departments.models.user import User
from departments.tasks import process_clinical_chatbot_task
from extensions import db


@pytest.fixture
def logged_in_user(client, app):
    with app.app_context():
        user = User.query.filter_by(username="doc_async_test").first()
        if not user:
            user = User(
                username="doc_async_test",
                password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
                role="admin",
            )
            db.session.add(user)
            db.session.commit()

    client.post(
        "/login",
        data={"username": "doc_async_test", "password": "Password123!"},
    )
    return user


def test_chatbot_async_route_dispatches_task_immediately(client, logged_in_user):
    """
    Verify POST /medicine/chatbot dispatches task via .delay() and returns HTTP 202 immediately.
    """
    mock_async_res = MagicMock()
    mock_async_res.id = "mock-task-id-12345"

    with patch.object(
        process_clinical_chatbot_task, "delay", return_value=mock_async_res
    ) as mock_delay:
        resp = client.post(
            "/medicine/chatbot",
            data={"clinical_note": "Patient presents with persistent cough and mild fever."},
        )

        assert resp.status_code == 202
        json_data = resp.get_json()
        assert json_data["status"] == "PROCESSING"
        assert json_data["task_id"] == "mock-task-id-12345"
        assert "/medicine/chatbot/status/mock-task-id-12345" in json_data["status_url"]

        # Verify .delay() was called with prompt text
        mock_delay.assert_called_once()
        args, _ = mock_delay.call_args
        assert "persistent cough and mild fever" in args[0]


def test_chatbot_celery_task_execution(app):
    """
    Verify process_clinical_chatbot_task executes AI summarization and returns structured result.
    """
    with app.app_context():
        mock_html = "<p><b>Clinical Assessment:</b> Acute Bronchitis likely.</p>"

        with patch(
            "departments.tasks.UniversalClinicalSummarizer.answer",
            return_value=mock_html,
        ) as mock_answer:
            # Execute underlying task function directly (simulating Celery task execution)
            task_func = getattr(process_clinical_chatbot_task, "__wrapped__", process_clinical_chatbot_task)
            result = task_func(
                "Patient presents with fever and cough",
                conversation_context=[],
                patient_id=None,
            )

            mock_answer.assert_called_once()
            assert result["status"] == "SUCCESS"
            assert "Acute Bronchitis" in result["summary_html"]
            assert result["raw_text"] == "Clinical Assessment: Acute Bronchitis likely."


def test_chatbot_polling_status_endpoint(client, logged_in_user):
    """
    Verify GET /medicine/chatbot/status/<task_id> returns task status without throwing DisabledBackend or 500 error.
    """
    mock_async_res = MagicMock()
    mock_async_res.state = "SUCCESS"
    mock_async_res.result = {
        "status": "SUCCESS",
        "summary_html": "<div>AI Summary Result</div>",
        "raw_text": "AI Summary Result",
        "input_note": "Fever test note",
    }

    with patch("departments.medicine.chat_bot.AsyncResult", return_value=mock_async_res):
        resp = client.get("/medicine/chatbot/status/test-task-999")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "SUCCESS"
        assert data["task_id"] == "test-task-999"
        assert data["summary_html"] == "<div>AI Summary Result</div>"
