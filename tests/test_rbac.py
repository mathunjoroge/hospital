import pytest

from app import app
from departments.models.user import User
from extensions import db


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        with app.test_client() as client:
            yield client
        db.session.remove()
        db.drop_all()

def test_unauthorized_role_returns_403(client):
    """Verify that accessing admin routes with non-admin role returns 403 Forbidden."""
    # Create test user in DB
    user = User(id=999, username='nurse_user', password='password', role='nursing')
    db.session.add(user)
    db.session.commit()

    with client.session_transaction() as sess:
        sess['_user_id'] = '999'
        sess['_fresh'] = True

    resp = client.get('/admin/')
    assert resp.status_code == 403

