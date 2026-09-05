import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            yield client

def test_chat_question_xss_autoescaped(client):
    """Verify that user questions in chat_bot.html are HTML escaped, preventing stored XSS."""
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
        sess['_fresh'] = True

    payload = "<script>alert('xss')</script>"
    # Render template with payload in conversation
    with app.test_request_context():
        from flask import render_template
        rendered = render_template('medicine/chat_bot.html',
                                   conversation=[{'question': payload, 'response': 'Hello\nWorld'}],
                                   form=None, clear_form=None)
        assert "<script>alert('xss')</script>" not in rendered
        assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in rendered or "&lt;script&gt;" in rendered

def test_nl2br_safe_filter(client):
    """Verify that nl2br_safe escapes HTML tags while converting newlines to <br>."""
    with app.app_context():
        from flask import render_template_string
        template = "{{ payload | nl2br_safe }}"
        result = render_template_string(template, payload="<script>alert(1)</script>\nLine 2")
        assert "<script>" not in result
        assert "&lt;script&gt;alert(1)&lt;/script&gt;<br>Line 2" in result
