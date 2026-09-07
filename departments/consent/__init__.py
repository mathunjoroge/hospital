from flask import Blueprint

bp = Blueprint("consent", __name__, url_prefix="/consent")

from . import routes  # noqa: E402, F401
