import pytest

from app import app
from departments.models.user import User
from extensions import db


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        db.create_all()
        with app.test_client() as client:
            yield client
        db.session.remove()
        db.drop_all()


def test_unauthorized_role_returns_403(client):
    """Verify that accessing admin routes with non-admin role returns 403 Forbidden."""
    user = User(id=999, username="nurse_user", password="password", role="nursing")
    db.session.add(user)
    db.session.commit()

    with client.session_transaction() as sess:
        sess["_user_id"] = "999"
        sess["_fresh"] = True

    resp = client.get("/admin/")
    assert resp.status_code == 403


def test_admin_switched_user_role_enforcement(client):
    """
    Verify that when an admin switches role (e.g. to nursing), the effective role
    restricts their access to non-nursing routes (e.g. 403 on admin-only route),
    while an admin with no active switch retains full access.
    """
    admin_user = User(id=1000, username="admin_user", password="password", role="admin")
    db.session.add(admin_user)
    db.session.commit()

    # 1. Admin without role switch -> Full access (200 OK)
    with client.session_transaction() as sess:
        sess["_user_id"] = "1000"
        sess["_fresh"] = True
        sess.pop("switched_user", None)

    resp_normal = client.get("/admin/")
    assert resp_normal.status_code == 200, f"Expected 200 for un-switched admin, got {resp_normal.status_code}"

    # 2. Admin with switched_user = 'nursing' -> Restricted from admin route (403 Forbidden)
    with client.session_transaction() as sess:
        sess["_user_id"] = "1000"
        sess["_fresh"] = True
        sess["switched_user"] = "nursing"

    resp_switched = client.get("/admin/")
    assert resp_switched.status_code == 403, f"Expected 403 for admin switched to nursing, got {resp_switched.status_code}"


def test_admin_revert_user_route(client):
    """
    Verify that an admin in a switched role state can access /admin/revert_user
    to clear switched_user and successfully revert back to full admin access.
    """
    admin_user = User(id=1001, username="revert_admin", password="password", role="admin")
    db.session.add(admin_user)
    db.session.commit()

    with client.session_transaction() as sess:
        sess["_user_id"] = "1001"
        sess["_fresh"] = True
        sess["switched_user"] = "nursing"

    # Revert user route must be accessible without 403 Forbidden
    resp = client.get("/admin/revert_user", follow_redirects=True)
    assert resp.status_code == 200

    # Verify switched_user session key was popped
    with client.session_transaction() as sess:
        assert "switched_user" not in sess
