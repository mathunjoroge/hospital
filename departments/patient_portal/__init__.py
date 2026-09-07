from flask import Blueprint

patient_portal_bp = Blueprint("patient_portal", __name__, template_folder="templates")

from . import auth, routes  # noqa: F401, E402
