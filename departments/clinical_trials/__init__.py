from flask import Blueprint

bp = Blueprint("clinical_trials", __name__, url_prefix="/clinical-trials")

from . import routes  # noqa: F401, E402
