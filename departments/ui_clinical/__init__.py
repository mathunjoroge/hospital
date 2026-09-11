from flask import Blueprint

bp = Blueprint("ui_clinical", __name__, url_prefix="/ui/clinical", template_folder="templates")
from . import routes  # noqa: F401
