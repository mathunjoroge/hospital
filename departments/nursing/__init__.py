from flask import Blueprint

bp = Blueprint("nursing", __name__, template_folder="templates")

from . import care_tasks, notes, vitals  # noqa: F401
