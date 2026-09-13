from flask import Blueprint

bp = Blueprint("laboratory", __name__, template_folder="templates")

from . import lims_routes, reagents, results, tests  # noqa: F401
