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
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = os.getenv('REDIS_PORT', '6379')
    redis_pass = os.getenv('REDIS_PASSWORD')
    if redis_pass:
        default_redis = f'redis://:{redis_pass}@{redis_host}:{redis_port}/0'
    else:
        default_redis = f'redis://{redis_host}:{redis_port}/0'

    redis_url = os.getenv('REDIS_URL', default_redis)

    celery_instance = Celery(
        flask_app.import_name,
        backend=redis_url,
        broker=redis_url
    )
    celery_instance.conf.update(
        result_backend=redis_url,
        broker_url=redis_url,
        result_expires=3600,  # 1 hour expiration for task results
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True,
    )
    celery_instance.set_default()

    class ContextTask(celery_instance.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery_instance.Task = ContextTask
    return celery_instance

celery = make_celery(app)
import departments.tasks  # noqa: F401
