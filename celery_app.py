"""
celery_app.py
─────────────
Celery application factory and instance configuration.
Connects Celery tasks to the Flask application context using Redis as the broker.
"""

import os

from celery import Celery

from app import app


def make_celery(flask_app):
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    celery_instance = Celery(
        flask_app.import_name,
        backend=redis_url,
        broker=redis_url
    )
    celery_instance.conf.update(
        result_backend=redis_url,
        broker_url=redis_url,
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True,
    )

    class ContextTask(celery_instance.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery_instance.Task = ContextTask
    return celery_instance

celery = make_celery(app)
