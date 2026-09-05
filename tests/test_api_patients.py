import pytest
from app import app
from extensions import db
from departments.models.user import User

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

def test_patient_search_unauthenticated(client):
    """Verify that unauthenticated access is redirected to login (302) or 401."""
    resp = client.get('/api/patients/search?q=test')
    assert resp.status_code in [302, 401]

def test_patient_search_authenticated(client):
    """Verify that authenticated access returns 200."""
    user = User(id=999, username='test_user', password='password', role='admin')
    db.session.add(user)
    db.session.commit()

    with client.session_transaction() as sess:
        sess['_user_id'] = '999'
        sess['_fresh'] = True

    resp = client.get('/api/patients/search?q=test')
    assert resp.status_code == 200
